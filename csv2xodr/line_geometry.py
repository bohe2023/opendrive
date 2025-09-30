"""Utilities for parsing white line geometry from CSV inputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from csv2xodr.normalize.core import latlon_to_local_xy
from csv2xodr.simpletable import DataFrame
from csv2xodr.topology.core import _canonical_numeric


def _find_column(columns: Iterable[str], *keywords: str) -> Optional[str]:
    lowered = [kw.lower() for kw in keywords]
    for col in columns:
        stripped = col.strip()
        value = stripped.lower()
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


def _geometry_signature(points: List[Tuple[float, float, float, float]]) -> Tuple[Tuple[int, int, int, int], ...]:
    return tuple(
        (
            int(round(s * 1000)),
            int(round(x * 1000)),
            int(round(y * 1000)),
            int(round(z * 1000)),
        )
        for s, x, y, z in points
    )


def build_line_geometry_lookup(
    line_geom_df: Optional[DataFrame],
    *,
    offset_mapper=None,
    lat0: float,
    lon0: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return a lookup of canonical line IDs to polylines in local XY."""

    if line_geom_df is None or len(line_geom_df) == 0:
        return {}

    cols = list(line_geom_df.columns)

    line_id_col = (
        _find_column(cols, "ライン", "ID")
        or _find_column(cols, "区画線", "ID")
        or _find_column(cols, "lane", "line", "id")
    )
    lat_col = _find_column(cols, "緯度") or _find_column(cols, "Latitude")
    lon_col = _find_column(cols, "経度") or _find_column(cols, "Longitude")
    z_col = _find_column(cols, "高さ") or _find_column(cols, "標高") or _find_column(cols, "Height")
    offset_col = None
    for col in cols:
        stripped = col.strip().lower()
        if "offset" in stripped and "end" not in stripped:
            offset_col = col
            break
    end_offset_col = _find_column(cols, "End", "Offset")
    type_col = _find_column(cols, "Type") or _find_column(cols, "種別")
    logtime_col = _find_column(cols, "ログ") or _find_column(cols, "log", "Time")
    instance_col = _find_column(cols, "Instance") or _find_column(cols, "インスタンス")
    flag_col = _find_column(cols, "3D") or _find_column(cols, "属性")
    is_retrans_col = _find_column(cols, "Is", "Retransmission")

    if line_id_col is None or lat_col is None or lon_col is None:
        return {}

    grouped: Dict[Tuple[str, Any, Any, Any, Any], Dict[str, Any]] = {}
    base_offsets: Dict[str, float] = {}

    for idx in range(len(line_geom_df)):
        row = line_geom_df.iloc[idx]

        line_id = _canonical_numeric(row[line_id_col], allow_negative=True)
        if line_id is None:
            continue

        lat_val = _to_float(row[lat_col])
        lon_val = _to_float(row[lon_col])
        if lat_val is None or lon_val is None:
            continue

        z_val = _to_float(row[z_col]) if z_col else 0.0
        if z_val is None:
            z_val = 0.0

        off_val = None
        if offset_col is not None:
            off_val = _to_float(row[offset_col])
        if off_val is None and end_offset_col is not None:
            off_val = _to_float(row[end_offset_col])

        if off_val is None:
            continue

        group_key = (
            line_id,
            row[logtime_col] if logtime_col else None,
            row[instance_col] if instance_col else None,
            row[flag_col] if flag_col else None,
            row[type_col] if type_col else None,
        )

        entry = grouped.setdefault(
            group_key,
            {
                "line_id": line_id,
                "lat": [],
                "lon": [],
                "z": [],
                "offset": [],
                "has_true": False,
                "has_false": False,
                "has_flag": False,
            },
        )

        retrans_flag = None
        if is_retrans_col is not None:
            entry["has_flag"] = True
            value = row[is_retrans_col]
            if isinstance(value, bool):
                retrans_flag = value
            else:
                retrans_flag = str(value).strip().lower() == "true"
        if retrans_flag is True:
            entry["has_true"] = True
        elif retrans_flag is False:
            entry["has_false"] = True

        off_m = off_val / 100.0
        current_base = base_offsets.get(line_id)
        if current_base is None or off_m < current_base:
            base_offsets[line_id] = off_m

        entry["lat"].append(lat_val)
        entry["lon"].append(lon_val)
        entry["z"].append(z_val)
        entry["offset"].append(off_m)

    if not grouped:
        return {}

    lookup: Dict[str, List[Dict[str, Any]]] = {}

    for entry in grouped.values():
        has_true = entry.get("has_true", False)
        has_false = entry.get("has_false", False)
        has_flag = entry.get("has_flag", False)
        if has_flag and has_true and not has_false:
            continue

        offsets_raw = entry.get("offset", [])
        if offsets_raw:
            base_offset = base_offsets.get(entry["line_id"], 0.0)
            offsets_m = [value - base_offset for value in offsets_raw]
        else:
            offsets_m = []
        if offset_mapper is not None:
            mapped_s = [offset_mapper(value) for value in offsets_m]
        else:
            mapped_s = offsets_m

        x_vals, y_vals = latlon_to_local_xy(entry["lat"], entry["lon"], lat0, lon0)

        points = list(zip(mapped_s, x_vals, y_vals, entry["z"]))
        if not points:
            continue

        signature = _geometry_signature(points)

        geoms = lookup.setdefault(entry["line_id"], [])
        if any(_geometry_signature(list(zip(g["s"], g["x"], g["y"], g["z"]))) == signature for g in geoms):
            continue

        geoms.append(
            {
                "s": mapped_s,
                "x": x_vals,
                "y": y_vals,
                "z": entry["z"],
            }
        )

    return lookup

