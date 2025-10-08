from xml.etree.ElementTree import Element, SubElement, tostring
import math
import xml.dom.minidom as minidom
from typing import Optional


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

def write_xodr(
    centerline,
    sections,
    lane_spec_per_section,
    out_path,
    geo_ref=None,
    elevation_profile=None,
    geometry_segments=None,
    superelevation_profile=None,
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
                    "hdg": _format_float(seg["hdg"], precision=15),
                    "length": _format_float(length, precision=12),
                },
            )
            curvature = float(seg.get("curvature", 0.0))
            if abs(curvature) > 1e-9:
                SubElement(geom, "arc", {"curvature": _format_float(curvature, precision=12)})
            else:
                SubElement(geom, "line")
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
                    "hdg": _format_float(hdg, precision=15),
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

    # lanes
    lanes = SubElement(road, "lanes")
    for sec in lane_spec_per_section:
        attrs = {"s": _format_float(sec["s0"], precision=9)}
        has_left = bool(sec.get("left"))
        has_right = bool(sec.get("right"))
        if has_left != has_right:
            attrs["singleSide"] = "true"

        ls = SubElement(lanes, "laneSection", attrs)

        center_el = SubElement(ls, "center")
        SubElement(center_el, "lane", {"id": "0", "type": "none", "level": "false"})

        left_el = SubElement(ls, "left") if has_left else None
        right_el = SubElement(ls, "right") if has_right else None

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
                    s_vals = geometry.get("s") or []
                    x_vals = geometry.get("x") or []
                    y_vals = geometry.get("y") or []
                    z_vals = geometry.get("z") or []
                    curvature_vals = geometry.get("curvature") or []
                    has_curvature = len(curvature_vals) == len(s_vals)

                    if (
                        len(s_vals) == len(x_vals)
                        and len(s_vals) == len(y_vals)
                        and len(s_vals) == len(z_vals)
                        and len(s_vals) >= 2
                    ):
                        if not explicit_geometry_written:
                            explicit_geometry_written = True
                            header.set("revMinor", "6")
                            header.set("version", "1.06")

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
                            if has_curvature:
                                try:
                                    raw_curv = curvature_vals[idx]
                                    curvature_val = float(raw_curv) if raw_curv is not None else None
                                except (TypeError, ValueError):
                                    curvature_val = None

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
                                    "hdg": _format_float(arc_heading, precision=15),
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
                                "hdg": _format_float(hdg, precision=15),
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

        for lane_data in sec.get("left", []):
            if left_el is None:
                left_el = SubElement(ls, "left")
            _write_lane(left_el, lane_data)

        for lane_data in sec.get("right", []):
            if right_el is None:
                right_el = SubElement(ls, "right")
            _write_lane(right_el, lane_data)

    with open(out_path, "wb") as f:
        f.write(_pretty(odr))
    return out_path
