from xml.etree.ElementTree import Element, SubElement, tostring
import math
import xml.dom.minidom as minidom
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _format_float(value: float, precision: int = 9) -> str:
    formatted = f"{float(value):.{precision}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


def _pretty(elem: Element) -> bytes:
    rough = tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")


def _solve_heading_for_arc(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    length: float,
    curvature: float,
) -> Optional[float]:
    if not math.isfinite(curvature) or abs(curvature) <= 1e-12:
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    if not math.isfinite(length) or length <= 1e-9:
        return None

    dx_target = x1 - x0
    dy_target = y1 - y0
    if not (math.isfinite(dx_target) and math.isfinite(dy_target)):
        return None

    hdg = math.atan2(dy_target, dx_target)
    max_iter = 16
    for _ in range(max_iter):
        end_hdg = hdg + curvature * length
        sin_start = math.sin(hdg)
        cos_start = math.cos(hdg)
        sin_end = math.sin(end_hdg)
        cos_end = math.cos(end_hdg)

        dx = (sin_end - sin_start) / curvature
        dy = (-cos_end + cos_start) / curvature
        err_x = dx_target - dx
        err_y = dy_target - dy
        err = math.hypot(err_x, err_y)
        if err <= 1e-6:
            return hdg

        ddx = (cos_end - cos_start) / curvature
        ddy = (sin_end - sin_start) / curvature
        denom = ddx * ddx + ddy * ddy
        if denom <= 1e-18 or not math.isfinite(denom):
            break

        delta = (err_x * ddx + err_y * ddy) / denom
        if not math.isfinite(delta):
            break

        if abs(delta) > math.pi:
            delta = math.copysign(math.pi, delta)

        hdg += delta
        if abs(delta) <= 1e-9:
            break

    end_hdg = hdg + curvature * length
    dx = (math.sin(end_hdg) - math.sin(hdg)) / curvature
    dy = (-math.cos(end_hdg) + math.cos(hdg)) / curvature
    err = math.hypot(dx_target - dx, dy_target - dy)
    if err <= 1e-4:
        return hdg
    return None


def _build_param_poly3_segment(
    s0: float,
    s1: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    tx0: float,
    ty0: float,
    tx1: float,
    ty1: float,
    curvature_start: Optional[float] = None,
    curvature_end: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    length = s1 - s0
    if not math.isfinite(length) or length <= 1e-6:
        return None

    start_len = math.hypot(tx0, ty0)
    if start_len <= 1e-12:
        dx = x1 - x0
        dy = y1 - y0
        chord = math.hypot(dx, dy)
        if chord <= 1e-12:
            return None
        tx0, ty0 = dx / chord, dy / chord
        start_len = 1.0
    else:
        tx0 /= start_len
        ty0 /= start_len

    end_len = math.hypot(tx1, ty1)
    if end_len <= 1e-12:
        dx = x1 - x0
        dy = y1 - y0
        chord = math.hypot(dx, dy)
        if chord <= 1e-12:
            return None
        tx1, ty1 = dx / chord, dy / chord
    else:
        tx1 /= end_len
        ty1 /= end_len

    heading = math.atan2(ty0, tx0)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    dx = x1 - x0
    dy = y1 - y0
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None

    # Rotate into the start-frame so the initial tangent aligns with the +x axis.
    u1 = cos_h * dx + sin_h * dy
    v1 = -sin_h * dx + cos_h * dy

    tx1_local = cos_h * tx1 + sin_h * ty1
    ty1_local = -sin_h * tx1 + cos_h * ty1

    norm_end = math.hypot(tx1_local, ty1_local)
    if norm_end <= 1e-12:
        tx1_local, ty1_local = 1.0, 0.0
    else:
        tx1_local /= norm_end
        ty1_local /= norm_end

    cos_delta = max(-1.0, min(1.0, tx1_local))
    sin_delta = max(-1.0, min(1.0, ty1_local))

    L = float(length)
    if not math.isfinite(L) or L <= 1e-9:
        return None

    denom = L ** 3
    if not math.isfinite(denom) or abs(denom) <= 1e-18:
        return None

    d1 = cos_delta - 1.0
    d2 = u1 - L
    d_u = (d1 * L - 2.0 * d2) / denom
    c_u = (d1 - 3.0 * d_u * L * L) / (2.0 * L)

    d_v = (sin_delta * L - 2.0 * v1) / denom
    c_v = (sin_delta - 3.0 * d_v * L * L) / (2.0 * L)

    if not all(
        math.isfinite(val)
        for val in (heading, c_u, d_u, c_v, d_v, u1, v1, cos_delta, sin_delta)
    ):
        return None

    spiral_curvature_start: Optional[float]
    spiral_curvature_end: Optional[float]
    spiral_curvature_start = None
    spiral_curvature_end = None
    if curvature_start is not None and curvature_end is not None:
        try:
            spiral_curvature_start = float(curvature_start)
            spiral_curvature_end = float(curvature_end)
        except (TypeError, ValueError):
            spiral_curvature_start = None
            spiral_curvature_end = None

    if (
        spiral_curvature_start is not None
        and spiral_curvature_end is not None
        and math.isfinite(spiral_curvature_start)
        and math.isfinite(spiral_curvature_end)
    ):
        if abs(spiral_curvature_end - spiral_curvature_start) >= 1e-4:
            return {
                "type": "spiral",
                "length": L,
                "heading": heading,
                "curvStart": spiral_curvature_start,
                "curvEnd": spiral_curvature_end,
            }

    return {
        "type": "paramPoly3",
        "length": L,
        "heading": heading,
        "aU": 0.0,
        "bU": 1.0,
        "cU": c_u,
        "dU": d_u,
        "aV": 0.0,
        "bV": 0.0,
        "cV": c_v,
        "dV": d_v,
    }


def _prepare_explicit_geometry(
    s_vals: Sequence[float],
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    z_vals: Sequence[float],
    tangent_x_vals: Sequence[float],
    tangent_y_vals: Sequence[float],
    curvature_vals: Sequence[float],
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    bool,
    bool,
]:
    def _safe_to_float_list(values: Sequence[float]) -> List[float]:
        result: List[float] = []
        for val in values:
            if val is None:
                result.append(math.nan)
                continue
            try:
                result.append(float(val))
            except (TypeError, ValueError):
                result.append(math.nan)
        return result

    try:
        import numpy as np  # type: ignore
    except ImportError:
        return (
            _safe_to_float_list(s_vals),
            _safe_to_float_list(x_vals),
            _safe_to_float_list(y_vals),
            _safe_to_float_list(z_vals),
            _safe_to_float_list(tangent_x_vals),
            _safe_to_float_list(tangent_y_vals),
            _safe_to_float_list(curvature_vals),
            len(tangent_x_vals) == len(s_vals) == len(tangent_y_vals),
            len(curvature_vals) == len(s_vals),
        )

    try:
        from scipy.signal import savgol_filter  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        savgol_filter = None  # type: ignore

    try:
        s_np = np.asarray(s_vals, dtype=float)
        x_np = np.asarray(x_vals, dtype=float)
        y_np = np.asarray(y_vals, dtype=float)
        z_np = np.asarray(z_vals, dtype=float)
    except (TypeError, ValueError):
        return (
            _safe_to_float_list(s_vals),
            _safe_to_float_list(x_vals),
            _safe_to_float_list(y_vals),
            _safe_to_float_list(z_vals),
            _safe_to_float_list(tangent_x_vals),
            _safe_to_float_list(tangent_y_vals),
            _safe_to_float_list(curvature_vals),
            len(tangent_x_vals) == len(s_vals) == len(tangent_y_vals),
            len(curvature_vals) == len(s_vals),
        )

    if s_np.size < 4:
        return (
            _safe_to_float_list(s_vals),
            _safe_to_float_list(x_vals),
            _safe_to_float_list(y_vals),
            _safe_to_float_list(z_vals),
            _safe_to_float_list(tangent_x_vals),
            _safe_to_float_list(tangent_y_vals),
            _safe_to_float_list(curvature_vals),
            len(tangent_x_vals) == len(s_vals) == len(tangent_y_vals),
            len(curvature_vals) == len(s_vals),
        )

    finite_mask = (
        np.isfinite(s_np)
        & np.isfinite(x_np)
        & np.isfinite(y_np)
        & np.isfinite(z_np)
    )

    tx_np = ty_np = None
    has_tangent = len(tangent_x_vals) == len(s_vals) == len(tangent_y_vals)
    if has_tangent:
        try:
            tx_np = np.asarray(tangent_x_vals, dtype=float)
            ty_np = np.asarray(tangent_y_vals, dtype=float)
            finite_mask &= np.isfinite(tx_np) & np.isfinite(ty_np)
        except (TypeError, ValueError):
            tx_np = ty_np = None
            has_tangent = False

    k_np = None
    has_curvature = len(curvature_vals) == len(s_vals)
    if has_curvature:
        try:
            k_np = np.asarray(curvature_vals, dtype=float)
            finite_mask &= np.isfinite(k_np)
        except (TypeError, ValueError):
            k_np = None
            has_curvature = False

    if np.count_nonzero(finite_mask) < 4:
        return (
            _safe_to_float_list(s_vals),
            _safe_to_float_list(x_vals),
            _safe_to_float_list(y_vals),
            _safe_to_float_list(z_vals),
            _safe_to_float_list(tangent_x_vals),
            _safe_to_float_list(tangent_y_vals),
            _safe_to_float_list(curvature_vals),
            has_tangent,
            has_curvature,
        )

    s_np = s_np[finite_mask]
    x_np = x_np[finite_mask]
    y_np = y_np[finite_mask]
    z_np = z_np[finite_mask]
    if tx_np is not None and ty_np is not None:
        tx_np = tx_np[finite_mask]
        ty_np = ty_np[finite_mask]
    if k_np is not None:
        k_np = k_np[finite_mask]

    diffs = np.diff(s_np)
    keep = np.concatenate(([True], diffs > 5e-4))
    s_np = s_np[keep]
    x_np = x_np[keep]
    y_np = y_np[keep]
    z_np = z_np[keep]
    if tx_np is not None and ty_np is not None:
        tx_np = tx_np[keep]
        ty_np = ty_np[keep]
    if k_np is not None:
        k_np = k_np[keep]

    if s_np.size < 4:
        return (
            s_np.tolist(),
            x_np.tolist(),
            y_np.tolist(),
            z_np.tolist(),
            tx_np.tolist() if tx_np is not None else _safe_to_float_list(tangent_x_vals),
            ty_np.tolist() if ty_np is not None else _safe_to_float_list(tangent_y_vals),
            k_np.tolist() if k_np is not None else _safe_to_float_list(curvature_vals),
            tx_np is not None and ty_np is not None,
            k_np is not None,
        )

    cut_values: List[float] = []
    if k_np is not None and k_np.size >= 3:
        k_s = k_np
        if savgol_filter is not None and k_s.size >= 5:
            window = min(max(int(k_s.size // 2 * 2 + 1), 5), 15)
            try:
                k_s = savgol_filter(k_s, window_length=window, polyorder=3, mode="interp")
            except ValueError:
                pass
        try:
            k_diff = np.abs(np.gradient(k_s, s_np))
        except Exception:
            k_diff = np.zeros_like(k_s)
        cut_mask = (k_diff > 1e-4) & (np.abs(k_s) > 1e-3)
        cut_values = [float(s_np[idx]) for idx in np.flatnonzero(cut_mask)]

    target_spacing = 5.0
    new_s: List[float] = [float(s_np[0])]
    anchors = sorted(set(cut_values + [float(s_np[-1])]))
    last = new_s[0]
    for anchor in anchors:
        if anchor <= new_s[-1] + 1e-4:
            continue
        while anchor - last > target_spacing:
            last += target_spacing
            if anchor - last <= 1e-4:
                break
            new_s.append(last)
        if anchor - new_s[-1] > 1e-4:
            new_s.append(anchor)
        last = new_s[-1]

    if new_s[-1] < float(s_np[-1]) - 1e-4:
        new_s.append(float(s_np[-1]))

    s_new = np.asarray(new_s, dtype=float)
    s_new.sort()
    s_new = np.unique(s_new)
    if s_new.size < 2:
        return (
            s_np.tolist(),
            x_np.tolist(),
            y_np.tolist(),
            z_np.tolist(),
            tx_np.tolist() if tx_np is not None else _safe_to_float_list(tangent_x_vals),
            ty_np.tolist() if ty_np is not None else _safe_to_float_list(tangent_y_vals),
            k_np.tolist() if k_np is not None else _safe_to_float_list(curvature_vals),
            tx_np is not None and ty_np is not None,
            k_np is not None,
        )

    x_new = np.interp(s_new, s_np, x_np)
    y_new = np.interp(s_new, s_np, y_np)
    z_new = np.interp(s_new, s_np, z_np)

    edge_order = 2 if s_new.size > 2 else 1
    dx_ds = np.gradient(x_new, s_new, edge_order=edge_order)
    dy_ds = np.gradient(y_new, s_new, edge_order=edge_order)
    tangent_norm = np.hypot(dx_ds, dy_ds)
    tangent_norm[tangent_norm <= 1e-9] = 1.0
    tx_new = dx_ds / tangent_norm
    ty_new = dy_ds / tangent_norm

    curvature_new = None
    if s_new.size >= 3:
        ddx_ds = np.gradient(dx_ds, s_new, edge_order=edge_order)
        ddy_ds = np.gradient(dy_ds, s_new, edge_order=edge_order)
        denom = tangent_norm ** 3
        curvature_new = np.zeros_like(dx_ds)
        valid = denom > 1e-9
        curvature_new[valid] = (
            (dx_ds[valid] * ddy_ds[valid] - dy_ds[valid] * ddx_ds[valid]) / denom[valid]
        )
        curvature_new[~valid] = 0.0
        if savgol_filter is not None and curvature_new.size >= 5:
            window = min(max(int(curvature_new.size // 2 * 2 + 1), 5), 15)
            try:
                curvature_new = savgol_filter(
                    curvature_new, window_length=window, polyorder=3, mode="interp"
                )
            except ValueError:
                pass
    elif k_np is not None:
        curvature_new = np.interp(s_new, s_np, k_np)

    return (
        s_new.tolist(),
        x_new.tolist(),
        y_new.tolist(),
        z_new.tolist(),
        tx_new.tolist(),
        ty_new.tolist(),
        curvature_new.tolist() if curvature_new is not None else [],
        True,
        curvature_new is not None,
    )

def write_xodr(
    centerline,
    sections,
    lane_spec_per_section,
    out_path,
    geo_ref=None,
    elevation_profile=None,
    geometry_segments=None,
    superelevation_profile=None,
    signals=None,
    objects=None,
    road_metadata=None,
):
    road_metadata = road_metadata or {}

    # root + header
    odr = Element("OpenDRIVE")
    header = SubElement(odr, "header", {
        "revMajor": "1",
        "revMinor": "4",
        "name": "csv2xodr",
        "version": "1.4",
        "date": "2025-09-16",
    })
    explicit_geometry_written = False
    if geo_ref:
        SubElement(header, "geoReference").text = geo_ref

    # single road
    length = float(centerline["s"].iloc[-1])
    road = SubElement(
        odr,
        "road",
        {
            "name": "road_1",
            "length": _format_float(length, precision=9),
            "id": "1",
            "junction": "-1",
        },
    )

    road_type = str(road_metadata.get("type", "town"))
    type_el = SubElement(road, "type", {"s": "0.0", "type": road_type})

    speed = road_metadata.get("speed")
    if speed is not None:
        if isinstance(speed, (int, float)):
            speed_attrs = {"max": _format_float(speed), "unit": "m/s"}
        elif isinstance(speed, dict):
            speed_attrs = {str(k): str(v) for k, v in speed.items()}
            if "max" in speed_attrs:
                try:
                    speed_attrs["max"] = _format_float(float(speed_attrs["max"]))
                except (TypeError, ValueError):
                    pass
        else:
            raise TypeError("speed must be a number or a mapping of attributes")

        SubElement(type_el, "speed", speed_attrs)

    # planView with piecewise lines
    plan = SubElement(road, "planView")
    if geometry_segments:
        for seg in geometry_segments:
            try:
                length = float(seg.get("length", 0.0))
            except (TypeError, ValueError):
                continue

            if not math.isfinite(length) or length <= 1e-6:
                # 参考线在某些数据集中会出现零长度的占位段，如果仍然写入
                # OpenDRIVE 会被查看器当成新的起点，造成路段出现错位。这里
                # 直接丢弃这类节点，保持几何连续。
                continue

            geom = SubElement(
                plan,
                "geometry",
                {

                    # planView 节点之间必须在数值上无缝衔接，否则在 OpenDRIVE
                    # 查看器中会出现肉眼可见的缝隙。这里将几何参数输出为更高
                    # 的小数精度，以保留 build_geometry_segments 中累积积分的
                    # 结果，避免由于字符串截断导致的断裂。
                    "s": _format_float(seg["s"], precision=12),
                    "x": _format_float(seg["x"], precision=12),
                    "y": _format_float(seg["y"], precision=12),
                    "hdg": _format_float(seg["hdg"], precision=17),
                    "length": _format_float(length, precision=12),
                },
            )
            curvature = float(seg.get("curvature", 0.0))
            curvature_start = float(seg.get("curvature_start", curvature))
            curvature_end = float(seg.get("curvature_end", curvature))
            curvature_tol = 5e-5
            if (
                abs(curvature_start - curvature_end) <= curvature_tol
                and abs(curvature - curvature_start) <= curvature_tol
            ):
                if abs(curvature) > 1e-9:
                    SubElement(
                        geom,
                        "arc",
                        {"curvature": _format_float(curvature, precision=12)},
                    )
                else:
                    SubElement(geom, "line")
            else:
                SubElement(
                    geom,
                    "spiral",
                    {
                        "curvatureStart": _format_float(curvature_start, precision=12),
                        "curvatureEnd": _format_float(curvature_end, precision=12),
                    },
                )
    else:
        for i in range(len(centerline) - 1):
            s = float(centerline["s"].iloc[i])
            x = float(centerline["x"].iloc[i])
            y = float(centerline["y"].iloc[i])
            hdg = float(centerline["hdg"].iloc[i])
            x2 = float(centerline["x"].iloc[i + 1])
            y2 = float(centerline["y"].iloc[i + 1])
            seg_len = ((x2 - x) ** 2 + (y2 - y) ** 2) ** 0.5
            if not math.isfinite(seg_len) or seg_len <= 1e-6:
                # 如果中心线中存在重复点，会导出零长度几何段，从而在
                # OpenDRIVE 查看器中显示为断裂。忽略这些异常点，保证
                # 前后几何段首尾相接。
                continue
            geom = SubElement(
                plan,
                "geometry",
                {
                    # 与圆弧模式相同，采用更高的小数精度输出折线节点，
                    # 避免由于字符串截断导致的相邻路段起终点坐标不完全
                    # 一致，从而在查看器中出现细微豁口。
                    "s": _format_float(s, precision=12),
                    "x": _format_float(x, precision=12),
                    "y": _format_float(y, precision=12),
                    "hdg": _format_float(hdg, precision=17),
                    "length": _format_float(seg_len, precision=12),
                },
            )
            SubElement(geom, "line")

    if elevation_profile:
        elev = SubElement(road, "elevationProfile")
        for entry in elevation_profile:
            attrs = {
                "s": f"{entry['s']:.3f}",
                "a": _format_float(entry.get("a", 0.0)),
                "b": _format_float(entry.get("b", 0.0)),
                "c": _format_float(entry.get("c", 0.0)),
                "d": _format_float(entry.get("d", 0.0)),
            }
            SubElement(elev, "elevation", attrs)

    if superelevation_profile:
        lateral = SubElement(road, "lateralProfile")
        for entry in superelevation_profile:
            attrs = {
                "s": f"{entry['s']:.3f}",
                "a": _format_float(entry.get("a", 0.0)),
                "b": _format_float(entry.get("b", 0.0)),
                "c": _format_float(entry.get("c", 0.0)),
                "d": _format_float(entry.get("d", 0.0)),
            }
            SubElement(lateral, "superelevation", attrs)

    if objects:
        objects_el = SubElement(road, "objects")
        for obj in objects:
            attrs = {
                "id": str(obj.get("id", "")),
                "name": str(obj.get("name", "")),
                "type": str(obj.get("type", "")),
                "s": _format_float(obj.get("s", 0.0), precision=9),
                "t": _format_float(obj.get("t", 0.0), precision=9),
            }

            for key in ("orientation",):
                val = obj.get(key)
                if val is not None:
                    attrs[key] = str(val)

            for key in ("zOffset", "hdg", "pitch", "roll", "length", "width", "height"):
                val = obj.get(key)
                if val is None:
                    continue
                attrs[key] = _format_float(float(val))

            SubElement(objects_el, "object", attrs)

    if signals:
        signals_el = SubElement(road, "signals")
        for signal in signals:
            attrs = {
                "s": _format_float(signal.get("s", 0.0), precision=9),
                "t": _format_float(signal.get("t", 0.0), precision=9),
                "id": str(signal.get("id", "")),
            }

            value = signal.get("value")
            if value is not None:
                if isinstance(value, (int, float)):
                    attrs["value"] = _format_float(value)
                else:
                    attrs["value"] = str(value)

            z_offset = signal.get("zOffset")
            if z_offset is not None:
                attrs["zOffset"] = _format_float(float(z_offset))

            for key in (
                "name",
                "type",
                "subtype",
                "unit",
                "dynamic",
                "orientation",
                "country",
                "supplementary",
                "shape",
                "height",
                "width",
            ):
                val = signal.get(key)
                if val is None:
                    continue
                attrs[key] = str(val)

            SubElement(signals_el, "signal", attrs)

    # lanes
    lanes = SubElement(road, "lanes")

    lane_offsets: List[Tuple[float, float]] = []
    for sec in lane_spec_per_section:
        offset_val = sec.get("laneOffset")
        if offset_val is None:
            continue
        try:
            offset_float = float(offset_val)
        except (TypeError, ValueError):
            continue
        if abs(offset_float) <= 1e-6:
            continue
        try:
            s_pos = float(sec.get("s0", 0.0))
        except (TypeError, ValueError):
            s_pos = 0.0
        lane_offsets.append((s_pos, offset_float))

    if lane_offsets:
        lane_offsets.sort(key=lambda item: item[0])
        last_written: Optional[float] = None
        for s_pos, offset_float in lane_offsets:
            if last_written is not None and abs(offset_float - last_written) <= 1e-6:
                continue
            SubElement(
                lanes,
                "laneOffset",
                {
                    "s": _format_float(s_pos, precision=9),
                    "a": _format_float(offset_float),
                    "b": "0",
                    "c": "0",
                    "d": "0",
                },
            )
            last_written = offset_float

    for sec in lane_spec_per_section:
        attrs = {"s": _format_float(sec["s0"], precision=9)}

        raw_left = list(sec.get("left") or [])
        raw_right = list(sec.get("right") or [])

        center_lanes: List[Dict[str, Any]] = []

        def _partition_center(lanes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            filtered: List[Dict[str, Any]] = []
            for lane in lanes:
                lane_no = lane.get("lane_no")
                lane_no_val: Optional[float] = None
                if lane_no is not None:
                    try:
                        lane_no_val = float(lane_no)
                    except (TypeError, ValueError):
                        lane_no_val = None
                if lane_no_val is not None and abs(lane_no_val) <= 0.5:
                    center_lanes.append(lane)
                else:
                    filtered.append(lane)
            return filtered

        section_left_lanes = _partition_center(raw_left)
        section_right_lanes = _partition_center(raw_right)

        has_left = bool(section_left_lanes)
        has_right = bool(section_right_lanes)
        if has_left != has_right:
            attrs["singleSide"] = "true"

        ls = SubElement(lanes, "laneSection", attrs)

        section_s0 = float(sec["s0"])

        def _write_lane(parent, lane_data):
            nonlocal explicit_geometry_written
            lane_id = lane_data["id"]
            lane_type = lane_data.get("type", "driving")
            ln = SubElement(
                parent,
                "lane",
                {
                    "id": str(lane_id),
                    "type": lane_type,
                    "level": str(lane_data.get("level", "false")).lower(),
                },
            )
            width = float(lane_data.get("width", 3.5))
            SubElement(ln, "width", {"sOffset": "0.0", "a": f"{width:.3f}", "b": "0", "c": "0", "d": "0"})
            road_mark = lane_data.get("roadMark")
            if road_mark is None and lane_type != "shoulder":
                road_mark = {"type": "solid", "width": 0.12, "laneChange": "both"}

            if road_mark is not None:
                rm_attrs = {
                    "sOffset": "0.0",
                    "type": str(road_mark.get("type", "solid")),
                    "weight": str(road_mark.get("weight", "standard")),
                    "width": f"{float(road_mark.get('width', 0.12)):.3f}",
                    "color": str(road_mark.get("color", "standard")),
                    "laneChange": str(road_mark.get("laneChange", "both")),
                }
                rm_el = SubElement(ln, "roadMark", rm_attrs)

                geometry = None
                if isinstance(road_mark, dict):
                    geometry = road_mark.get("geometry")

                if geometry:
                    raw_s_vals = geometry.get("s") or []
                    raw_x_vals = geometry.get("x") or []
                    raw_y_vals = geometry.get("y") or []
                    raw_z_vals = geometry.get("z") or []
                    raw_curvature_vals = geometry.get("curvature") or []
                    raw_tangent_x_vals = geometry.get("tangent_x") or []
                    raw_tangent_y_vals = geometry.get("tangent_y") or []

                    (
                        s_vals,
                        x_vals,
                        y_vals,
                        z_vals,
                        tangent_x_vals,
                        tangent_y_vals,
                        curvature_vals,
                        has_tangent,
                        has_curvature,
                    ) = _prepare_explicit_geometry(
                        raw_s_vals,
                        raw_x_vals,
                        raw_y_vals,
                        raw_z_vals,
                        raw_tangent_x_vals,
                        raw_tangent_y_vals,
                        raw_curvature_vals,
                    )

                    if has_curvature and len(curvature_vals) != len(s_vals):
                        has_curvature = False

                    if has_tangent and len(tangent_x_vals) != len(s_vals):
                        has_tangent = False

                    if (
                        len(s_vals) == len(x_vals)
                        and len(s_vals) == len(y_vals)
                        and len(s_vals) == len(z_vals)
                        and len(s_vals) >= 2
                    ):
                        if not explicit_geometry_written:
                            explicit_geometry_written = True
                            header.set("revMinor", "4")
                            header.set("version", "1.4")

                        explicit_el = SubElement(rm_el, "explicit")

                        for idx in range(len(s_vals) - 1):
                            try:
                                s0 = float(s_vals[idx])
                                s1 = float(s_vals[idx + 1])
                                x0 = float(x_vals[idx])
                                x1 = float(x_vals[idx + 1])
                                y0 = float(y_vals[idx])
                                y1 = float(y_vals[idx + 1])
                                z0 = float(z_vals[idx])
                            except (TypeError, ValueError):
                                continue

                            segment_length = s1 - s0
                            if not math.isfinite(segment_length) or segment_length <= 1e-9:
                                continue

                            curvature_val: Optional[float] = None
                            curvature_val_next: Optional[float] = None
                            if has_curvature:
                                try:
                                    raw_curv = curvature_vals[idx]
                                    curvature_val = float(raw_curv) if raw_curv is not None else None
                                    raw_curv_next = curvature_vals[idx + 1]
                                    curvature_val_next = (
                                        float(raw_curv_next) if raw_curv_next is not None else None
                                    )
                                except (TypeError, ValueError):
                                    curvature_val = None
                                    curvature_val_next = None
                                else:
                                    if curvature_val is not None and not math.isfinite(curvature_val):
                                        curvature_val = None
                                    if (
                                        curvature_val_next is not None
                                        and not math.isfinite(curvature_val_next)
                                    ):
                                        curvature_val_next = None

                            param_poly: Optional[Dict[str, float]] = None
                            if has_tangent:
                                try:
                                    tx0 = float(tangent_x_vals[idx])
                                    ty0 = float(tangent_y_vals[idx])
                                    tx1 = float(tangent_x_vals[idx + 1])
                                    ty1 = float(tangent_y_vals[idx + 1])
                                    param_poly = _build_param_poly3_segment(
                                        float(s_vals[idx]),
                                        float(s_vals[idx + 1]),
                                        x0,
                                        y0,
                                        x1,
                                        y1,
                                        tx0,
                                        ty0,
                                        tx1,
                                        ty1,
                                        curvature_val,
                                        curvature_val_next,
                                    )
                                except (TypeError, ValueError):
                                    param_poly = None

                            if param_poly is not None:
                                geom_attrs = {
                                    "sOffset": _format_float(s0 - section_s0, precision=12),
                                    "x": _format_float(x0, precision=12),
                                    "y": _format_float(y0, precision=12),
                                    "z": _format_float(z0, precision=12),
                                    "hdg": _format_float(param_poly["heading"], precision=17),
                                    "length": _format_float(param_poly["length"], precision=12),
                                }

                                geom_el = SubElement(explicit_el, "geometry", geom_attrs)
                                if param_poly.get("type") == "spiral":
                                    SubElement(
                                        geom_el,
                                        "spiral",
                                        {
                                            "curvStart": _format_float(
                                                param_poly["curvStart"], precision=12
                                            ),
                                            "curvEnd": _format_float(
                                                param_poly["curvEnd"], precision=12
                                            ),
                                        },
                                    )
                                else:
                                    SubElement(
                                        geom_el,
                                        "paramPoly3",
                                        {
                                            "aU": _format_float(param_poly["aU"], precision=12),
                                            "bU": _format_float(param_poly["bU"], precision=12),
                                            "cU": _format_float(param_poly["cU"], precision=12),
                                            "dU": _format_float(param_poly["dU"], precision=12),
                                            "aV": _format_float(param_poly["aV"], precision=12),
                                            "bV": _format_float(param_poly["bV"], precision=12),
                                            "cV": _format_float(param_poly["cV"], precision=12),
                                            "dV": _format_float(param_poly["dV"], precision=12),
                                            "pRange": "arcLength",
                                        },
                                    )
                                continue

                            arc_heading = None
                            if curvature_val is not None and abs(curvature_val) > 1e-12:
                                arc_heading = _solve_heading_for_arc(
                                    x0, y0, x1, y1, segment_length, curvature_val
                                )

                            if arc_heading is not None:
                                geom_attrs = {
                                    "sOffset": _format_float(s0 - section_s0, precision=12),
                                    "x": _format_float(x0, precision=12),
                                    "y": _format_float(y0, precision=12),
                                    "z": _format_float(z0, precision=12),
                                    "hdg": _format_float(arc_heading, precision=17),
                                    "length": _format_float(segment_length, precision=12),
                                }

                                geom_el = SubElement(explicit_el, "geometry", geom_attrs)
                                SubElement(
                                    geom_el,
                                    "arc",
                                    {"curvature": _format_float(curvature_val, precision=12)},
                                )
                                continue

                            chord_length = math.hypot(x1 - x0, y1 - y0)
                            if not math.isfinite(chord_length) or chord_length <= 1e-6:
                                continue

                            hdg = math.atan2(y1 - y0, x1 - x0)

                            geom_attrs = {
                                "sOffset": _format_float(s0 - section_s0, precision=12),
                                "x": _format_float(x0, precision=12),
                                "y": _format_float(y0, precision=12),
                                "z": _format_float(z0, precision=12),
                                "hdg": _format_float(hdg, precision=17),
                                "length": _format_float(chord_length, precision=12),
                            }

                            geom_el = SubElement(explicit_el, "geometry", geom_attrs)

                            # Use a simple line primitive when curvature data is unavailable.
                            SubElement(geom_el, "line")

            predecessors = lane_data.get("predecessors") or []
            successors = lane_data.get("successors") or []
            if predecessors or successors:
                link = SubElement(ln, "link")
                for pid in predecessors:
                    SubElement(link, "predecessor", {"id": str(pid)})
                for sid in successors:
                    SubElement(link, "successor", {"id": str(sid)})

        center_el = SubElement(ls, "center")

        center_lane_data: Optional[Dict[str, Any]] = None
        if center_lanes:
            # Prefer a central lane that carries an explicit width definition.
            center_lane_data = max(
                center_lanes,
                key=lambda item: float(item.get("width", 0.0) or 0.0),
            )

        if center_lane_data is not None:
            promoted = dict(center_lane_data)
            promoted["id"] = 0
            promoted.setdefault("type", "driving")
            promoted["lane_no"] = 0
            promoted["predecessors"] = []
            promoted["successors"] = []
            _write_lane(center_el, promoted)
        else:
            fallback_lane = SubElement(
                center_el, "lane", {"id": "0", "type": "none", "level": "false"}
            )
            # MATLAB 的 Driving Scenario Designer 需要中心车道提供宽度信息来推导道路截面。
            # 当源数据缺少中心车道时，为导入器写入一条零宽度的占位 lane，避免报出
            # “找不到道路定义”的错误。
            SubElement(
                fallback_lane,
                "width",
                {
                    "sOffset": "0.0",
                    "a": _format_float(0.0, precision=3),
                    "b": "0",
                    "c": "0",
                    "d": "0",
                },
            )

        left_el = SubElement(ls, "left") if has_left else None
        right_el = SubElement(ls, "right") if has_right else None

        for lane_data in section_left_lanes:
            if left_el is None:
                left_el = SubElement(ls, "left")
            _write_lane(left_el, lane_data)

        for lane_data in section_right_lanes:
            if right_el is None:
                right_el = SubElement(ls, "right")
            _write_lane(right_el, lane_data)

    with open(out_path, "wb") as f:
        f.write(_pretty(odr))
    return out_path
