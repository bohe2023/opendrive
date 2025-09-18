import math
from typing import Iterable, List, Tuple

from csv2xodr.simpletable import DataFrame

def _col_like(df: DataFrame, keyword: str):
    cols = [c for c in df.columns if keyword in c]
    return cols[0] if cols else None

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

    if offsets is not None and len(offsets) == len(s):
        offsets_f = [float(v) for v in offsets]
        if offsets_f:
            start = offsets_f[0]
            offsets_norm = [v - start for v in offsets_f]
            if s[-1] > 0 and offsets_norm[-1] > 0:
                ratio = offsets_norm[-1] / s[-1]
                if ratio > 10.0:
                    offsets_norm = [v * 0.01 for v in offsets_norm]
            s = offsets_norm

    center = DataFrame({"s": s, "x": x, "y": y, "hdg": hdg})
    return center, (lat0, lon0)
