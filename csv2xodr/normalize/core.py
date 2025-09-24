import math
from typing import Callable, Iterable, List, Tuple, Optional

from csv2xodr.simpletable import DataFrame

def _col_like(df: DataFrame, keyword: str):
    cols = [c for c in df.columns if keyword in c]
    return cols[0] if cols else None


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

        if best_path is not None and row[path_col] != best_path:
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

    points: List[Tuple[float, float]] = []
    for offset_cm in sorted(grouped.keys()):
        heights = grouped[offset_cm]
        if not heights:
            continue
        avg_height = sum(heights) / len(heights)
        offset_m = offset_cm * 0.01
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
        elif profile:
            slope = profile[-1]["b"]
        else:
            slope = 0.0

        profile.append({
            "s": s_val,
            "a": height,
            "b": slope,
            "c": 0.0,
            "d": 0.0,
        })

    return profile
