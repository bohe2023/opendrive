import math
import numpy as np
import pandas as pd

def _col_like(df: pd.DataFrame, keyword: str):
    cols = [c for c in df.columns if keyword in c]
    return cols[0] if cols else None

def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Simple equirectangular projection to local XY [m].
    lat/lon can be numpy arrays in degrees.
    """
    R = 6378137.0
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    lat0 = math.radians(lat0)
    lon0 = math.radians(lon0)
    x = (lon - lon0) * np.cos((lat + lat0) / 2.0) * R
    y = (lat - lat0) * R
    return x, y

def build_centerline(df_line_geo: pd.DataFrame, df_base: pd.DataFrame):
    """
    Build centerline planView from PROFILETYPE_MPU_LINE_GEOMETRY (lat/lon series).
    Returns: centerline DataFrame [s,x,y,hdg], and (lat0, lon0)
    """
    if df_line_geo is None or len(df_line_geo) == 0:
        raise ValueError("line_geometry CSV is required")

    # choose origin
    if df_base is not None and len(df_base) > 0:
        lat0 = float(df_base.filter(like="緯度").iloc[0, 0])
        lon0 = float(df_base.filter(like="経度").iloc[0, 0])
    else:
        lat0 = float(df_line_geo.filter(like="緯度").mean(numeric_only=True))
        lon0 = float(df_line_geo.filter(like="経度").mean(numeric_only=True))

    lat_col = df_line_geo.filter(like="緯度").columns[0]
    lon_col = df_line_geo.filter(like="経度").columns[0]

    lat = df_line_geo[lat_col].astype(float).to_numpy()
    lon = df_line_geo[lon_col].astype(float).to_numpy()

    x, y = latlon_to_local_xy(lat, lon, lat0, lon0)

    # cumulative s & heading
    s = np.zeros_like(x)
    hdg = np.zeros_like(x)
    for i in range(1, len(x)):
        dx, dy = (x[i] - x[i - 1]), (y[i] - y[i - 1])
        ds = math.hypot(dx, dy)
        s[i] = s[i - 1] + ds
        hdg[i - 1] = math.atan2(dy, dx)
    if len(hdg) > 1:
        hdg[-1] = hdg[-2]
    center = pd.DataFrame({"s": s, "x": x, "y": y, "hdg": hdg})
    return center, (lat0, lon0)
