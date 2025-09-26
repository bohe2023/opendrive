import math
from typing import Callable, Iterable, List, Tuple, Optional, Any, Dict

from csv2xodr.simpletable import DataFrame

def _col_like(df: DataFrame, keyword: str):
    cols = [c for c in df.columns if keyword in c]
    return cols[0] if cols else None


def _find_column(df: DataFrame, *keywords: str, exclude: Tuple[str, ...] = ()) -> Optional[str]:
    lowered = [kw.lower() for kw in keywords]
    excluded = [kw.lower() for kw in exclude]
    for col in df.columns:
        stripped = col.strip()
        value = stripped.lower()
        if any(block in value for block in excluded):
            continue
        if all(keyword in value for keyword in lowered):
            return col
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return None
        return float(text)
    except Exception:  # pragma: no cover - defensive
        return None


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _find_height_column(df: DataFrame) -> Optional[str]:
    for col in df.columns:
        stripped = col.strip()
        lowered = stripped.lower()
        if "高さ" in stripped or "標高" in stripped or "height" in lowered or "[m]" in stripped:
            return col
    return None

def latlon_to_local_xy(lat: Iterable[float], lon: Iterable[float], lat0: float, lon0: float) -> Tuple[List[float], List[float]]:
    """
    Simple equirectangular projection to local XY [m].
    lat/lon can be numpy arrays in degrees.
    """
    R = 6378137.0
    lat_rad = [math.radians(v) for v in lat]
    lon_rad = [math.radians(v) for v in lon]
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x_vals = [
        (lon_v - lon0_rad) * math.cos((lat_v + lat0_rad) / 2.0) * R
        for lat_v, lon_v in zip(lat_rad, lon_rad)
    ]
    y_vals = [(lat_v - lat0_rad) * R for lat_v in lat_rad]
    return x_vals, y_vals

def build_centerline(df_line_geo: DataFrame, df_base: DataFrame):
    """
    Build centerline planView from PROFILETYPE_MPU_LINE_GEOMETRY (lat/lon series).
    Returns: centerline DataFrame [s,x,y,hdg], and (lat0, lon0)
    """
    if df_line_geo is None or len(df_line_geo) == 0:
        raise ValueError("line_geometry CSV is required")

    lat_col = df_line_geo.filter(like="緯度").columns[0]
    lon_col = df_line_geo.filter(like="経度").columns[0]

    offset_col = _col_like(df_line_geo, "Offset")
    if offset_col is not None:
        offsets_series = df_line_geo[offset_col].astype(float)
    else:
        offsets_series = None

    # Some providers emit one row per lane boundary for the same longitudinal
    # offset.  Averaging the duplicated offsets keeps the geometry focused on a
    # single centerline instead of weaving across multiple boundaries (which
    # renders as a cross/diamond artifact in the exported OpenDRIVE).
    if offsets_series is not None and offsets_series.duplicated().any():
        grouped = df_line_geo.groupby(offset_col, sort=True)[[lat_col, lon_col]].mean().reset_index()
        df_line_geo = grouped
        lat = [float(v) for v in grouped[lat_col].to_list()]
        lon = [float(v) for v in grouped[lon_col].to_list()]
        offsets = [float(v) for v in grouped[offset_col].to_list()]
    else:
        lat = [float(v) for v in df_line_geo[lat_col].astype(float).to_list()]
        lon = [float(v) for v in df_line_geo[lon_col].astype(float).to_list()]
        offsets = offsets_series.to_list() if offsets_series is not None else None

    # Some datasets interleave multiple Path Id polylines. Stick to the longest one to
    # avoid creating self-crossing planViews.
    path_col = _col_like(df_line_geo, "Path")
    if path_col is not None and df_line_geo[path_col].nunique(dropna=True) > 1:
        approx_lat0 = float(lat[0])
        approx_lon0 = float(lon[0])
        x_tmp, y_tmp = latlon_to_local_xy(lat, lon, approx_lat0, approx_lon0)
        lengths = {}
        path_series = df_line_geo[path_col]
        for pid in path_series.dropna().unique():
            indices = [i for i, value in enumerate(path_series.to_list()) if value == pid]
            if len(indices) < 2:
                continue
            ds_total = 0.0
            for i in range(len(indices) - 1):
                a = indices[i]
                b = indices[i + 1]
                dx = x_tmp[b] - x_tmp[a]
                dy = y_tmp[b] - y_tmp[a]
                ds_total += math.hypot(dx, dy)
            if ds_total > 0:
                lengths[pid] = ds_total
        if lengths:
            best_pid = max(lengths, key=lengths.get)
        else:
            best_pid = path_series.dropna().iloc[0]
        keep_mask = [value == best_pid for value in path_series.to_list()]
        df_line_geo = df_line_geo.loc[keep_mask].reset_index(drop=True)
        lat = [value for value, keep in zip(lat, keep_mask) if keep]
        lon = [value for value, keep in zip(lon, keep_mask) if keep]

    # choose origin
    if df_base is not None and len(df_base) > 0:
        lat0 = float(df_base.filter(like="緯度").iloc[0, 0])
        lon0 = float(df_base.filter(like="経度").iloc[0, 0])
    else:
        lat0 = sum(lat) / len(lat)
        lon0 = sum(lon) / len(lon)

    x, y = latlon_to_local_xy(lat, lon, lat0, lon0)

    # cumulative s & heading
    s = [0.0 for _ in x]
    hdg = [0.0 for _ in x]
    for i in range(1, len(x)):
        dx, dy = (x[i] - x[i - 1]), (y[i] - y[i - 1])
        ds = math.hypot(dx, dy)
        s[i] = s[i - 1] + ds
        hdg[i - 1] = math.atan2(dy, dx)
    if len(hdg) > 1:
        hdg[-1] = hdg[-2]

    offsets_column = None
    if offsets is not None and len(offsets) == len(s):
        offsets_f = [float(v) for v in offsets]
        if offsets_f:
            start = offsets_f[0]
            offsets_norm = [v - start for v in offsets_f]
            if s[-1] > 0 and offsets_norm[-1] > 0:
                ratio = offsets_norm[-1] / s[-1]
                if ratio > 10.0:
                    offsets_norm = [v * 0.01 for v in offsets_norm]
            offsets_column = offsets_norm

    data = {"s": s, "x": x, "y": y, "hdg": hdg}
    if offsets_column is not None:
        data["s_offset"] = offsets_column

    center = DataFrame(data)
    return center, (lat0, lon0)


def build_offset_mapper(centerline: DataFrame) -> Callable[[float], float]:
    """Return a callable that maps CSV offsets to centreline arc-length."""

    if "s_offset" not in centerline.columns:
        return lambda value: float(value)

    offsets = [float(v) for v in centerline["s_offset"].to_list()]
    s_vals = [float(v) for v in centerline["s"].to_list()]

    if not offsets or len(offsets) != len(s_vals):
        return lambda value: float(value)

    def mapper(value: float) -> float:
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except Exception:  # pragma: no cover - defensive
                return s_vals[0]

        if value <= offsets[0]:
            return s_vals[0]

        for i in range(1, len(offsets)):
            lo = offsets[i - 1]
            hi = offsets[i]
            if value <= hi:
                if hi <= lo:
                    return s_vals[i]
                t = (value - lo) / (hi - lo)
                return s_vals[i - 1] + t * (s_vals[i] - s_vals[i - 1])

        return s_vals[-1]

    return mapper


def build_elevation_profile(
    df_line_geo: DataFrame,
    *,
    offset_mapper: Optional[Callable[[float], float]] = None,
) -> List[dict]:
    """Build an OpenDRIVE elevation profile from line geometry heights.

    The PROFILETYPE_MPU_LINE_GEOMETRY.csv source encodes longitudinal
    offsets in centimetres alongside absolute height values.  The
    resulting profile is emitted as a list of dictionaries ready to be
    serialised into ``<elevation>`` elements where ``a`` represents the
    height at ``s`` and ``b`` encodes the gradient (first derivative).
    Only the primary path is considered when multiple polylines are
    present in the input.
    """

    if df_line_geo is None or len(df_line_geo) == 0:
        return []

    offset_col = _col_like(df_line_geo, "Offset")
    height_col = _find_height_column(df_line_geo)
    if offset_col is None or height_col is None:
        return []

    path_col = _col_like(df_line_geo, "Path")
    retrans_col: Optional[str] = None
    for col in df_line_geo.columns:
        if "retrans" in col.lower():
            retrans_col = col
            break

    best_path = None
    if path_col is not None and df_line_geo[path_col].nunique(dropna=True) > 1:
        counts = {}
        path_series = df_line_geo[path_col]
        for value in path_series.to_list():
            if value is None:
                continue
            counts[value] = counts.get(value, 0) + 1
        if counts:
            best_path = max(counts, key=counts.get)

    grouped: dict = {}
    for idx in range(len(df_line_geo)):
        row = df_line_geo.iloc[idx]

        if retrans_col is not None:
            retrans_value = row[retrans_col]
            if isinstance(retrans_value, str):
                retrans_flag = retrans_value.strip().lower() == "true"
            else:
                retrans_flag = bool(retrans_value)
            if retrans_flag:
                continue

        if best_path is not None and path_col is not None and row[path_col] != best_path:
            continue

        offset_raw = row[offset_col]
        height_raw = row[height_col]
        try:
            offset_cm = float(offset_raw)
            height = float(height_raw)
        except (TypeError, ValueError):
            continue

        grouped.setdefault(offset_cm, []).append(height)

    if not grouped:
        return []

    origin_cm = min(grouped.keys())

    points: List[Tuple[float, float]] = []
    for offset_cm in sorted(grouped.keys()):
        heights = grouped[offset_cm]
        if not heights:
            continue
        avg_height = sum(heights) / len(heights)
        offset_m = max(0.0, (offset_cm - origin_cm) * 0.01)
        if offset_mapper is not None:
            s_val = float(offset_mapper(offset_m))
        else:
            s_val = float(offset_m)
        points.append((s_val, avg_height))

    if not points:
        return []

    profile: List[dict] = []
    for idx, (s_val, height) in enumerate(points):
        if idx < len(points) - 1:
            next_s, next_height = points[idx + 1]
            if next_s > s_val:
                slope = (next_height - height) / (next_s - s_val)
            else:
                slope = 0.0
        else:
            # The final elevation entry does not have a following point to
            # constrain its gradient.  Re-using the previous slope may cause
            # the profile to extrapolate aggressively which manifests as a
            # vertical spike at the end of the road.  Default to a flat
            # continuation instead.
            slope = 0.0

        profile.append({
            "s": s_val,
            "a": height,
            "b": slope,
            "c": 0.0,
            "d": 0.0,
        })

    return profile


def _select_best_path(df: DataFrame, path_col: Optional[str]) -> Optional[Any]:
    if path_col is None:
        return None

    counts: Dict[Any, int] = {}
    series = df[path_col]
    for value in series.to_list():
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1

    if not counts:
        return None

    return max(counts, key=counts.get)


def _prepare_segment_key(start_m: float, end_m: float) -> Tuple[float, float]:
    return (round(float(start_m), 4), round(float(end_m), 4))


def build_curvature_profile(
    df_curvature: Optional[DataFrame],
    *,
    offset_mapper: Optional[Callable[[float], float]] = None,
) -> List[Dict[str, float]]:
    if df_curvature is None or len(df_curvature) == 0:
        return []

    start_col = _find_column(df_curvature, "offset", exclude=("end",))
    end_col = _find_column(df_curvature, "end", "offset")
    curvature_col = (
        _find_column(df_curvature, "曲率", exclude=("レーン", "lane", "情報"))
        or _find_column(df_curvature, "曲率", "値")
        or _find_column(df_curvature, "曲率", "rad/m")
        or _find_column(df_curvature, "curvature", exclude=("lane", "count"))
        or _find_column(df_curvature, "curvature", "value")
        or _find_column(df_curvature, "curvature")
    )
    path_col = _find_column(df_curvature, "path")
    retrans_col = _find_column(df_curvature, "is", "retransmission")

    if start_col is None or end_col is None or curvature_col is None:
        return []

    best_path = _select_best_path(df_curvature, path_col)

    entries: List[Tuple[float, float, float]] = []
    origin_cm: Optional[float] = None

    for idx in range(len(df_curvature)):
        row = df_curvature.iloc[idx]

        if retrans_col is not None:
            retrans_val = str(row[retrans_col]).strip().lower()
            if retrans_val == "true":
                continue

        if best_path is not None and path_col is not None and row[path_col] != best_path:
            continue

        start_cm = _to_float(row[start_col])
        end_cm = _to_float(row[end_col])
        curvature_val = _to_float(row[curvature_col])

        if start_cm is None or end_cm is None or curvature_val is None:
            continue

        if origin_cm is None or start_cm < origin_cm:
            origin_cm = start_cm

        entries.append((start_cm, end_cm, curvature_val))

    if origin_cm is None:
        return []

    grouped: Dict[Tuple[float, float], Dict[str, List[float]]] = {}

    for start_cm, end_cm, curvature_val in entries:
        start_m = max(0.0, (start_cm - origin_cm) * 0.01)
        end_m = max(0.0, (end_cm - origin_cm) * 0.01)
        if end_m <= start_m:
            continue

        key = _prepare_segment_key(start_m, end_m)
        bucket = grouped.setdefault(key, {"curvature": [], "length": []})
        bucket["curvature"].append(curvature_val)
        bucket["length"].append(end_m - start_m)

    if not grouped:
        return []

    profile: List[Dict[str, float]] = []
    for (start_m, end_m), values in sorted(grouped.items(), key=lambda item: item[0]):
        curv_values = values["curvature"]
        if not curv_values:
            continue
        avg_curvature = sum(curv_values) / len(curv_values)
        s0 = float(offset_mapper(start_m)) if offset_mapper is not None else float(start_m)
        s1 = float(offset_mapper(end_m)) if offset_mapper is not None else float(end_m)
        if s1 <= s0:
            continue
        if values["length"]:
            avg_length = sum(values["length"]) / len(values["length"])
        else:
            avg_length = end_m - start_m
        if avg_length <= 0:
            continue
        span = s1 - s0
        if span <= 0:
            continue
        scale = avg_length / span
        profile.append({"s0": s0, "s1": s1, "curvature": avg_curvature * scale})

    return profile


def build_slope_profile(
    df_slope: Optional[DataFrame],
    *,
    offset_mapper: Optional[Callable[[float], float]] = None,
) -> Dict[str, List[Dict[str, float]]]:
    if df_slope is None or len(df_slope) == 0:
        return {"longitudinal": [], "superelevation": []}

    start_col = _find_column(df_slope, "offset", exclude=("end",))
    end_col = _find_column(df_slope, "end", "offset")
    slope_col = _find_column(df_slope, "縦断勾配") or _find_column(df_slope, "longitudinal", "%")
    cross_col = _find_column(df_slope, "横断勾配") or _find_column(df_slope, "cross", "%")
    path_col = _find_column(df_slope, "path")
    retrans_col = _find_column(df_slope, "is", "retransmission")

    if start_col is None or end_col is None:
        return {"longitudinal": [], "superelevation": []}

    best_path = _select_best_path(df_slope, path_col)

    entries: List[Tuple[float, float, Optional[float], Optional[float]]] = []
    origin_cm: Optional[float] = None

    for idx in range(len(df_slope)):
        row = df_slope.iloc[idx]

        if retrans_col is not None:
            retrans_val = str(row[retrans_col]).strip().lower()
            if retrans_val == "true":
                continue

        if best_path is not None and path_col is not None and row[path_col] != best_path:
            continue

        start_cm = _to_float(row[start_col])
        end_cm = _to_float(row[end_col])
        if start_cm is None or end_cm is None:
            continue

        if origin_cm is None or start_cm < origin_cm:
            origin_cm = start_cm

        grade_val = _to_float(row[slope_col]) if slope_col is not None else None
        cross_val = _to_float(row[cross_col]) if cross_col is not None else None

        entries.append((start_cm, end_cm, grade_val, cross_val))

    if origin_cm is None:
        return {"longitudinal": [], "superelevation": []}

    grouped: Dict[Tuple[float, float], Dict[str, List[float]]] = {}

    for start_cm, end_cm, grade_val, cross_val in entries:
        start_m = max(0.0, (start_cm - origin_cm) * 0.01)
        end_m = max(0.0, (end_cm - origin_cm) * 0.01)
        if end_m <= start_m:
            continue

        entry = grouped.setdefault(
            _prepare_segment_key(start_m, end_m),
            {"grade": [], "cross": [], "length": []},
        )

        if grade_val is not None:
            entry["grade"].append(grade_val * 0.01)

        if cross_val is not None:
            entry["cross"].append(cross_val * 0.01)

        entry["length"].append(end_m - start_m)

    longitudinal: List[Dict[str, float]] = []
    superelevation: List[Dict[str, float]] = []

    for (start_m, end_m), values in sorted(grouped.items(), key=lambda item: item[0]):
        s0 = float(offset_mapper(start_m)) if offset_mapper is not None else float(start_m)
        s1 = float(offset_mapper(end_m)) if offset_mapper is not None else float(end_m)
        if s1 <= s0:
            continue

        if values["grade"]:
            avg_grade = sum(values["grade"]) / len(values["grade"])
            segment_span = s1 - s0
            if segment_span <= 0:
                continue
            if values["length"]:
                avg_length = sum(values["length"]) / len(values["length"])
            else:
                avg_length = segment_span
            if avg_length > 0:
                scale = avg_length / segment_span
                avg_grade *= scale
            longitudinal.append({"s0": s0, "s1": s1, "grade": avg_grade})

        if values["cross"]:
            avg_cross = sum(values["cross"]) / len(values["cross"])
            superelevation.append({"s0": s0, "s1": s1, "angle": avg_cross})

    return {"longitudinal": longitudinal, "superelevation": superelevation}


def build_elevation_profile_from_slopes(
    segments: List[Dict[str, float]],
    *,
    initial_height: float = 0.0,
) -> List[Dict[str, float]]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: (item["s0"], item.get("s1", item["s0"])))

    profile: List[Dict[str, float]] = []
    height = float(initial_height)
    prev_end = ordered[0]["s0"]
    last_grade = ordered[0]["grade"]

    first_s0 = ordered[0]["s0"]
    profile.append({"s": first_s0, "a": height, "b": last_grade, "c": 0.0, "d": 0.0})

    prev_end = ordered[0].get("s1", first_s0)
    if prev_end > first_s0:
        height += last_grade * (prev_end - first_s0)

    for seg in ordered[1:]:
        s0 = seg["s0"]
        s1 = seg.get("s1", s0)
        grade = seg["grade"]

        if s0 > prev_end:
            height += last_grade * (s0 - prev_end)
            prev_end = s0
        elif s0 < prev_end:
            s0 = prev_end

        profile.append({"s": s0, "a": height, "b": grade, "c": 0.0, "d": 0.0})

        if s1 > prev_end:
            height += grade * (s1 - s0)
            prev_end = s1

        last_grade = grade

    return profile


def build_superelevation_profile(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: (item["s0"], item.get("s1", item["s0"])))
    profile: List[Dict[str, float]] = []
    for seg in ordered:
        profile.append({"s": seg["s0"], "a": seg["angle"], "b": 0.0, "c": 0.0, "d": 0.0})
    return profile


def _interpolate_centerline(centerline: DataFrame, target_s: float) -> Tuple[float, float, float]:
    s_vals = [float(v) for v in centerline["s"].to_list()]
    x_vals = [float(v) for v in centerline["x"].to_list()]
    y_vals = [float(v) for v in centerline["y"].to_list()]
    hdg_vals = [float(v) for v in centerline["hdg"].to_list()]

    if not s_vals:
        return 0.0, 0.0, 0.0

    if target_s <= s_vals[0]:
        return x_vals[0], y_vals[0], hdg_vals[0]

    for idx in range(1, len(s_vals)):
        s_prev = s_vals[idx - 1]
        s_curr = s_vals[idx]
        if target_s <= s_curr:
            span = s_curr - s_prev
            if span <= 0:
                return x_vals[idx], y_vals[idx], hdg_vals[idx]
            t = (target_s - s_prev) / span
            x = x_vals[idx - 1] + t * (x_vals[idx] - x_vals[idx - 1])
            y = y_vals[idx - 1] + t * (y_vals[idx] - y_vals[idx - 1])
            if abs(target_s - s_curr) <= 1e-6:
                if idx >= len(hdg_vals) - 1:
                    hdg = hdg_vals[-1]
                else:
                    hdg = hdg_vals[idx - 1]
            else:
                hdg = hdg_vals[idx - 1]
            return x, y, hdg

    return x_vals[-1], y_vals[-1], hdg_vals[-1]


def build_geometry_segments(
    centerline: DataFrame,
    curvature_segments: List[Dict[str, float]],
    *,
    max_endpoint_deviation: float = 0.5,
    max_segment_length: float = 2.0,
) -> List[Dict[str, float]]:
    if not curvature_segments:
        return []

    total_length = float(centerline["s"].iloc[-1]) if len(centerline) else 0.0
    if total_length <= 0:
        return []

    centerline_s = [float(v) for v in centerline["s"].to_list()]

    def _build_segments(max_len: float) -> Tuple[List[Dict[str, float]], float]:
        def _clamp(value: float) -> float:
            clamped = max(0.0, min(total_length, float(value)))
            # 标记点在浮点运算中会出现微小误差，这里统一取 9 位小数以便后续
            # 查找时能够复用相同的键，避免 densify 过程中产生的重复点无法被
            # 正确识别为同一个位置。
            return round(clamped, 9)

        anchor_points: Dict[float, bool] = {_clamp(0.0): True, _clamp(total_length): True}

        for value in centerline_s:
            anchor_points.setdefault(_clamp(value), True)

        if max_len <= 0:
            effective_len = 2.0
        else:
            effective_len = float(max_len)

        # 长曲率区段在积分时会累积轻微的横向偏移，进而在 OpenDRIVE
        # 查看器中表现为相邻路段之间出现细小豁口。为了在不牺牲曲线段
        # 的情况下压制误差，将曲率分段进一步按照参考线采样 densify。
        # 这里允许调用方传入更小的 densify 间距；如果输入的最大长度
        # 失效，则退回默认的 2 米。
        # densify 由后续针对每个主控制点区间的细分逻辑统一处理，这里无需
        # 额外插入中间节点。

        for seg in curvature_segments:
            s0 = _clamp(seg["s0"])
            s1 = _clamp(seg["s1"])
            anchor_points.setdefault(s0, True)
            anchor_points.setdefault(s1, True)

        ordered_points = sorted(anchor_points.keys())
        segments: List[Dict[str, float]] = []

        def _curvature_for_interval(start: float, end: float) -> float:
            mid = (start + end) / 2.0
            for seg in curvature_segments:
                if seg["s0"] <= mid <= seg["s1"]:
                    return float(seg["curvature"])
            return 0.0

        def _integrate_segment(
            start_x: float,
            start_y: float,
            start_hdg: float,
            curvature: float,
            length: float,
        ) -> Tuple[float, float, float]:
            if abs(curvature) <= 1e-12:
                end_x = start_x + length * math.cos(start_hdg)
                end_y = start_y + length * math.sin(start_hdg)
                end_hdg = start_hdg
            else:
                end_hdg = _normalize_angle(start_hdg + curvature * length)
                radius = 1.0 / curvature
                dx = radius * (math.sin(end_hdg) - math.sin(start_hdg))
                dy = -radius * (math.cos(end_hdg) - math.cos(start_hdg))
                end_x = start_x + dx
                end_y = start_y + dy
            return end_x, end_y, end_hdg

        def _segment_derivatives(
            start_hdg: float,
            curvature: float,
            length: float,
        ) -> Tuple[float, float, float]:
            if abs(curvature) <= 1e-8:
                half_l_sq = 0.5 * (length ** 2)
                dxdk = -half_l_sq * math.sin(start_hdg)
                dydk = half_l_sq * math.cos(start_hdg)
                return dxdk, dydk, length

            end_hdg = start_hdg + curvature * length
            sin_start = math.sin(start_hdg)
            cos_start = math.cos(start_hdg)
            sin_end = math.sin(end_hdg)
            cos_end = math.cos(end_hdg)
            k = curvature
            L = length
            numerator_x = sin_end - sin_start
            numerator_y = cos_end - cos_start
            dxdk = (cos_end * L * k - numerator_x) / (k * k)
            dydk = (sin_end * L * k + numerator_y) / (k * k)
            return dxdk, dydk, L

        def _refine_curvature(
            start_x: float,
            start_y: float,
            start_hdg: float,
            length: float,
            initial_curvature: float,
            target_x: float,
            target_y: float,
            target_hdg: float,
            preferred_curvature: float,
        ) -> Tuple[float, float, float, float, float]:
            curvature = float(initial_curvature)
            preferred_sign = 0.0
            if abs(preferred_curvature) > 1e-12:
                preferred_sign = math.copysign(1.0, preferred_curvature)
            weight_angle = 0.25
            for _ in range(8):
                end_x, end_y, end_hdg = _integrate_segment(start_x, start_y, start_hdg, curvature, length)
                err_x = target_x - end_x
                err_y = target_y - end_y
                err_theta = _normalize_angle(target_hdg - end_hdg)
                if math.hypot(err_x, err_y) <= 1e-4 and abs(err_theta) <= 1e-4:
                    break

                dxdk, dydk, dthdk = _segment_derivatives(start_hdg, curvature, length)
                denom = dxdk * dxdk + dydk * dydk + weight_angle * (dthdk * dthdk)
                if denom <= 1e-18 or not math.isfinite(denom):
                    break

                numer = err_x * dxdk + err_y * dydk + weight_angle * err_theta * dthdk
                if not math.isfinite(numer):
                    break

                delta = numer / denom
                if not math.isfinite(delta):
                    break

                max_step = 0.5 / max(1.0, length)
                if delta > max_step:
                    delta = max_step
                elif delta < -max_step:
                    delta = -max_step

                if abs(delta) <= 1e-12:
                    break

                next_curvature = curvature + delta
                if preferred_sign and next_curvature * preferred_sign < 0:
                    next_curvature = float(preferred_curvature)
                curvature = next_curvature

            end_x, end_y, end_hdg = _integrate_segment(start_x, start_y, start_hdg, curvature, length)
            endpoint_error = math.hypot(end_x - target_x, end_y - target_y)
            if preferred_sign and curvature * preferred_sign < 0:
                curvature = float(preferred_curvature)
                end_x, end_y, end_hdg = _integrate_segment(start_x, start_y, start_hdg, curvature, length)
                endpoint_error = math.hypot(end_x - target_x, end_y - target_y)
            return curvature, end_x, end_y, end_hdg, endpoint_error

        current_s = ordered_points[0]
        current_x, current_y, current_hdg = _interpolate_centerline(centerline, current_s)
        max_observed_endpoint_deviation = 0.0

        for idx in range(len(ordered_points) - 1):
            start = ordered_points[idx]
            end = ordered_points[idx + 1]
            length_total = end - start
            if length_total <= 1e-6:
                continue
            curvature_dataset = _curvature_for_interval(start, end)
            preferred_sign = 0.0
            if abs(curvature_dataset) > 1e-12:
                preferred_sign = math.copysign(1.0, curvature_dataset)

            target_x, target_y, target_hdg = _interpolate_centerline(centerline, end)

            delta_target = _normalize_angle(target_hdg - current_hdg)
            delta_dataset = curvature_dataset * length_total
            if abs(delta_target) > 1e-5 and length_total > 1e-6:
                curvature_guess = curvature_dataset + (delta_target - delta_dataset) / length_total
            else:
                curvature_guess = curvature_dataset

            if preferred_sign and curvature_guess * preferred_sign < 0:
                curvature_guess = curvature_dataset

            next_x, next_y, next_hdg = _integrate_segment(
                current_x, current_y, current_hdg, curvature_guess, length_total
            )
            endpoint_error = math.hypot(next_x - target_x, next_y - target_y)

            if endpoint_error > max_endpoint_deviation + 1e-9:
                refined_curvature, next_x, next_y, next_hdg, endpoint_error = _refine_curvature(
                    current_x,
                    current_y,
                    current_hdg,
                    length_total,
                    curvature_guess,
                    target_x,
                    target_y,
                    target_hdg,
                    curvature_dataset,
                )
            else:
                refined_curvature = curvature_guess

            if preferred_sign and refined_curvature * preferred_sign < 0:
                refined_curvature = float(curvature_dataset)
                next_x, next_y, next_hdg = _integrate_segment(
                    current_x, current_y, current_hdg, refined_curvature, length_total
                )
                endpoint_error = math.hypot(next_x - target_x, next_y - target_y)

            steps = max(1, int(math.ceil(length_total / effective_len)))
            step_length = length_total / steps
            seg_start_s = start
            seg_start_x = current_x
            seg_start_y = current_y
            seg_start_hdg = current_hdg

            for step in range(steps):
                seg_s = seg_start_s + step * step_length
                seg_x = seg_start_x
                seg_y = seg_start_y
                seg_hdg = seg_start_hdg

                segments.append(
                    {
                        "s": seg_s,
                        "x": seg_x,
                        "y": seg_y,
                        "hdg": seg_hdg,
                        "length": step_length,
                        "curvature": refined_curvature,
                    }
                )

                seg_start_x, seg_start_y, seg_start_hdg = _integrate_segment(
                    seg_x, seg_y, seg_hdg, refined_curvature, step_length
                )

            current_x, current_y, current_hdg = seg_start_x, seg_start_y, seg_start_hdg
            current_s = end

            if endpoint_error > max_observed_endpoint_deviation:
                max_observed_endpoint_deviation = endpoint_error

        return segments, max_observed_endpoint_deviation

    try:
        initial_len = float(max_segment_length)
    except (TypeError, ValueError):
        initial_len = 2.0

    if initial_len <= 0:
        initial_len = 2.0

    best_segments: List[Dict[str, float]] = []
    best_deviation = float("inf")

    def _record_best(candidate: List[Dict[str, float]], deviation_value: float) -> None:
        nonlocal best_segments, best_deviation
        if candidate and deviation_value < best_deviation:
            best_segments = candidate
            best_deviation = deviation_value

    segments, deviation = _build_segments(initial_len)
    if not segments:
        return []

    _record_best(segments, deviation)

    min_len = max(0.25, initial_len / 16.0)
    current_len = initial_len

    # 在实际数据中，个别曲率段可能因为原始测量噪声导致理论圆弧与
    # 折线终点存在略高于阈值的偏差。通过逐步缩小 densify 间距可以
    # 有效抑制误差，而无需完全回退到折线表达。
    while deviation > max_endpoint_deviation and current_len > min_len:
        current_len = max(min_len, current_len / 2.0)
        candidate_segments, candidate_deviation = _build_segments(current_len)
        if not candidate_segments:
            break

        segments = candidate_segments
        deviation = candidate_deviation
        _record_best(segments, deviation)

    if deviation > max_endpoint_deviation:
        return best_segments

    return segments


def build_shoulder_profile(
    df_shoulder: Optional[DataFrame],
    *,
    offset_mapper: Optional[Callable[[float], float]] = None,
) -> List[Dict[str, float]]:
    if df_shoulder is None or len(df_shoulder) == 0:
        return []

    start_col = _find_column(df_shoulder, "offset", exclude=("end",))
    end_col = _find_column(df_shoulder, "end", "offset")
    left_col = _find_column(df_shoulder, "左", "路肩") or _find_column(df_shoulder, "left")
    right_col = _find_column(df_shoulder, "右", "路肩") or _find_column(df_shoulder, "right")
    path_col = _find_column(df_shoulder, "path")
    retrans_col = _find_column(df_shoulder, "is", "retransmission")

    if start_col is None or end_col is None:
        return []

    best_path = _select_best_path(df_shoulder, path_col)

    entries: List[Tuple[float, float, Optional[float], Optional[float]]] = []
    origin_cm: Optional[float] = None

    for idx in range(len(df_shoulder)):
        row = df_shoulder.iloc[idx]

        if retrans_col is not None:
            retrans_val = str(row[retrans_col]).strip().lower()
            if retrans_val == "true":
                continue

        if best_path is not None and row[path_col] != best_path:
            continue

        start_cm = _to_float(row[start_col])
        end_cm = _to_float(row[end_col])
        if start_cm is None or end_cm is None:
            continue

        if origin_cm is None or start_cm < origin_cm:
            origin_cm = start_cm

        left_val = _to_float(row[left_col]) if left_col is not None else None
        right_val = _to_float(row[right_col]) if right_col is not None else None

        entries.append((start_cm, end_cm, left_val, right_val))

    if origin_cm is None:
        return []

    grouped: Dict[Tuple[float, float], Dict[str, List[float]]] = {}

    for start_cm, end_cm, left_val, right_val in entries:
        start_m = max(0.0, (start_cm - origin_cm) * 0.01)
        end_m = max(0.0, (end_cm - origin_cm) * 0.01)
        if end_m <= start_m:
            continue

        entry = grouped.setdefault(_prepare_segment_key(start_m, end_m), {"left": [], "right": []})

        if left_val is not None:
            entry["left"].append(left_val * 0.01)

        if right_val is not None:
            entry["right"].append(right_val * 0.01)

    profile: List[Dict[str, float]] = []
    for (start_m, end_m), values in sorted(grouped.items(), key=lambda item: item[0]):
        s0 = float(offset_mapper(start_m)) if offset_mapper is not None else float(start_m)
        s1 = float(offset_mapper(end_m)) if offset_mapper is not None else float(end_m)
        if s1 <= s0:
            continue

        left_width = sum(values["left"]) / len(values["left"]) if values["left"] else 0.0
        right_width = sum(values["right"]) / len(values["right"]) if values["right"] else 0.0

        profile.append({"s0": s0, "s1": s1, "left": left_width, "right": right_width})

    return profile
