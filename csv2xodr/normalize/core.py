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

    grouped: Dict[Tuple[float, float], List[float]] = {}

    for start_cm, end_cm, curvature_val in entries:
        start_m = max(0.0, (start_cm - origin_cm) * 0.01)
        end_m = max(0.0, (end_cm - origin_cm) * 0.01)
        if end_m <= start_m:
            continue

        key = _prepare_segment_key(start_m, end_m)
        grouped.setdefault(key, []).append(curvature_val)

    if not grouped:
        return []

    profile: List[Dict[str, float]] = []
    for (start_m, end_m), values in sorted(grouped.items(), key=lambda item: item[0]):
        if not values:
            continue
        avg_curvature = sum(values) / len(values)
        s0 = float(offset_mapper(start_m)) if offset_mapper is not None else float(start_m)
        s1 = float(offset_mapper(end_m)) if offset_mapper is not None else float(end_m)
        if s1 <= s0:
            continue
        profile.append({"s0": s0, "s1": s1, "curvature": avg_curvature})

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

        entry = grouped.setdefault(_prepare_segment_key(start_m, end_m), {"grade": [], "cross": []})

        if grade_val is not None:
            entry["grade"].append(grade_val * 0.01)

        if cross_val is not None:
            entry["cross"].append(cross_val * 0.01)

    longitudinal: List[Dict[str, float]] = []
    superelevation: List[Dict[str, float]] = []

    for (start_m, end_m), values in sorted(grouped.items(), key=lambda item: item[0]):
        s0 = float(offset_mapper(start_m)) if offset_mapper is not None else float(start_m)
        s1 = float(offset_mapper(end_m)) if offset_mapper is not None else float(end_m)
        if s1 <= s0:
            continue

        if values["grade"]:
            avg_grade = sum(values["grade"]) / len(values["grade"])
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
            if abs(target_s - s_curr) <= 1e-6 and idx < len(hdg_vals):
                hdg = hdg_vals[idx]
            else:
                hdg = hdg_vals[idx - 1]
            return x, y, hdg

    return x_vals[-1], y_vals[-1], hdg_vals[-1]


def build_geometry_segments(
    centerline: DataFrame,
    curvature_segments: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    if not curvature_segments:
        return []

    total_length = float(centerline["s"].iloc[-1]) if len(centerline) else 0.0
    if total_length <= 0:
        return []

    breakpoints = {0.0, total_length}
    for seg in curvature_segments:
        s0 = max(0.0, min(total_length, float(seg["s0"])) )
        s1 = max(0.0, min(total_length, float(seg["s1"])) )
        breakpoints.add(s0)
        breakpoints.add(s1)

    ordered_points = sorted(breakpoints)
    segments: List[Dict[str, float]] = []

    def _curvature_for_interval(start: float, end: float) -> float:
        mid = (start + end) / 2.0
        for seg in curvature_segments:
            if seg["s0"] <= mid <= seg["s1"]:
                return float(seg["curvature"])
        return 0.0

    for idx in range(len(ordered_points) - 1):
        start = ordered_points[idx]
        end = ordered_points[idx + 1]
        length = end - start
        if length <= 1e-6:
            continue
        curvature = _curvature_for_interval(start, end)
        x, y, hdg = _interpolate_centerline(centerline, start)
        segments.append(
            {
                "s": start,
                "x": x,
                "y": y,
                "hdg": hdg,
                "length": length,
                "curvature": curvature,
            }
        )

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
