"""Utilities for parsing white line geometry from CSV inputs."""

from __future__ import annotations

import math
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
    shape_count_col = (
        _find_column(cols, "形状", "要素")
        or _find_column(cols, "shape", "count")
        or _find_column(cols, "shape", "points")
    )

    if line_id_col is None or lat_col is None or lon_col is None:
        return {}

    grouped: Dict[Tuple[str, Any, Any, Any, Any], Dict[str, Any]] = {}
    segment_progress: Dict[
        Tuple[Any, Any, Any, Any, Any, float, Optional[float]], Dict[str, float]
    ] = {}

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

        off_cm_raw = None
        if offset_col is not None:
            off_cm_raw = _to_float(row[offset_col])
        if off_cm_raw is None and end_offset_col is not None:
            off_cm_raw = _to_float(row[end_offset_col])

        if off_cm_raw is None:
            continue

        end_cm_val = _to_float(row[end_offset_col]) if end_offset_col is not None else None

        segment_key = (
            line_id,
            row[logtime_col] if logtime_col else None,
            row[instance_col] if instance_col else None,
            row[flag_col] if flag_col else None,
            row[type_col] if type_col else None,
            float(off_cm_raw),
            float(end_cm_val) if end_cm_val is not None else None,
        )

        tracker = segment_progress.get(segment_key)
        if tracker is None:
            count_val: Optional[int] = None
            if shape_count_col is not None:
                try:
                    raw_value = row[shape_count_col]
                except Exception:  # pragma: no cover - defensive
                    raw_value = None
                raw_count = _to_float(raw_value)
                if raw_count is not None and raw_count > 0:
                    try:
                        count_val = int(round(raw_count))
                    except Exception:  # pragma: no cover - defensive
                        count_val = None
            step_val: Optional[float] = None
            if count_val is not None and count_val > 1 and end_cm_val is not None:
                step_val = (float(end_cm_val) - float(off_cm_raw)) / float(count_val - 1)
            tracker = {
                "index": 0,
                "count": count_val,
                "step": step_val,
                "start": float(off_cm_raw),
            }
            segment_progress[segment_key] = tracker

        index = int(tracker.get("index", 0))
        tracker["index"] = index + 1
        count_val = tracker.get("count")
        step_val = tracker.get("step")
        start_cm = tracker.get("start", float(off_cm_raw))

        if count_val is not None and tracker["index"] >= count_val:
            segment_progress.pop(segment_key, None)

        if step_val is not None and count_val is not None and count_val > 1:
            off_val = start_cm + step_val * index
        else:
            off_val = float(off_cm_raw)

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

        entry["lat"].append(lat_val)
        entry["lon"].append(lon_val)
        entry["z"].append(z_val)
        entry["offset"].append(off_m)

    if not grouped:
        return {}

    lookup: Dict[str, List[Dict[str, Any]]] = {}

    global_base_offset: Optional[float] = None
    for entry in grouped.values():
        for raw in entry.get("offset", []) or []:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if global_base_offset is None or value < global_base_offset:
                global_base_offset = value

    for entry in grouped.values():
        has_true = entry.get("has_true", False)
        has_false = entry.get("has_false", False)
        has_flag = entry.get("has_flag", False)
        if has_flag and has_true and not has_false:
            continue

        lat_vals = entry.get("lat", []) or []
        lon_vals = entry.get("lon", []) or []
        z_vals = entry.get("z", []) or []
        offsets_raw = entry.get("offset", []) or []

        if not lat_vals or not lon_vals or not z_vals or not offsets_raw:
            continue

        try:
            x_vals, y_vals = latlon_to_local_xy(lat_vals, lon_vals, lat0, lon0)
        except Exception:  # pragma: no cover - defensive
            continue

        if len(x_vals) != len(z_vals) or len(x_vals) != len(offsets_raw):
            continue

        # The Japanese datasets reuse the same ``line_id`` across multiple
        # carriageway segments.  Each time the offset restarts from zero the
        # geometry would jump back to the beginning of the alignment, producing
        # spaghetti-like lane lines in the viewer.  Split the sequence whenever
        # the longitudinal coordinate decreases noticeably so every emitted
        # polyline remains strictly monotonic along ``s``.
        sequences: List[List[Tuple[float, float, float, float]]] = []
        current: List[Tuple[float, float, float, float]] = []
        last_s: Optional[float] = None
        reset_threshold = 1e-4  # tolerate sub-millimetre jitter while catching real resets

        for raw_offset, x_val, y_val, z_val in zip(offsets_raw, x_vals, y_vals, z_vals):
            try:
                offset_m = float(raw_offset)
            except (TypeError, ValueError):
                continue

            if global_base_offset is not None and math.isfinite(offset_m):
                offset_m -= global_base_offset

            if offset_mapper is not None:
                try:
                    s_float = float(offset_mapper(offset_m))
                except Exception:  # pragma: no cover - defensive
                    continue
            else:
                s_float = float(offset_m)

            if not math.isfinite(s_float):
                continue

            try:
                x_float = float(x_val)
                y_float = float(y_val)
                z_float = float(z_val)
            except (TypeError, ValueError):
                continue

            if last_s is not None and s_float < last_s - reset_threshold:
                if len(current) >= 2:
                    sequences.append(current)
                current = []
                last_s = None

            if current:
                prev_s, prev_x, prev_y, prev_z = current[-1]
                if (
                    abs(s_float - prev_s) <= 1e-9
                    and abs(x_float - prev_x) <= 1e-9
                    and abs(y_float - prev_y) <= 1e-9
                    and abs(z_float - prev_z) <= 1e-9
                ):
                    # Skip duplicate samples that would otherwise collapse into
                    # zero-length segments after splitting.
                    last_s = s_float
                    continue

            current.append((s_float, x_float, y_float, z_float))
            last_s = s_float

        if len(current) >= 2:
            sequences.append(current)

        geoms = lookup.setdefault(entry["line_id"], [])

        for seq in sequences:
            points = list(seq)
            signature = _geometry_signature(points)
            if any(
                _geometry_signature(list(zip(g["s"], g["x"], g["y"], g["z"]))) == signature
                for g in geoms
            ):
                continue

            geoms.append(
                {
                    "s": [p[0] for p in points],
                    "x": [p[1] for p in points],
                    "y": [p[2] for p in points],
                    "z": [p[3] for p in points],
                }
            )

    return lookup

