"""Utilities for parsing white line geometry from CSV inputs."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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


def _normalise_key(value: Any) -> Optional[str]:
    canonical = _canonical_numeric(value, allow_negative=True)
    if canonical is not None:
        return canonical
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_line_geometry_lookup(
    line_geom_df: Optional[DataFrame],
    *,
    offset_mapper=None,
    lat0: float,
    lon0: float,
    curvature_samples: Optional[Iterable[Dict[str, Any]]] = None,
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
    path_col = _find_column(cols, "Path", "Id")
    lane_col = _find_column(cols, "Lane", "Number") or _find_column(cols, "Lane", "No")

    if line_id_col is None or lat_col is None or lon_col is None:
        return {}

    grouped: Dict[Tuple[str, Any, Any, Any, Any], Dict[str, Any]] = {}
    segment_progress: Dict[
        Tuple[Any, Any, Any, Any, Any, float, Optional[float]], Dict[str, float]
    ] = {}

    prepared_samples: List[Tuple[float, float, float]] = []
    lane_samples_xy: Dict[
        Tuple[Optional[str], Optional[str]], List[Tuple[float, float, float]]
    ] = {}
    curvature_by_index: Dict[
        Tuple[Optional[str], Optional[str]], Dict[int, List[Tuple[Optional[float], float]]]
    ] = {}
    curvature_by_offset: Dict[
        Tuple[Optional[str], Optional[str]], Dict[float, List[float]]
    ] = {}
    if curvature_samples:
        for sample in curvature_samples:
            path_key = _normalise_key(sample.get("path"))
            lane_key = _normalise_key(sample.get("lane"))

            curv_val: Optional[float]
            try:
                curv_val = float(sample.get("curvature"))
            except (TypeError, ValueError):
                curv_val = None

            idx_val: Optional[float]
            try:
                idx_val = float(sample.get("shape_index"))
            except (TypeError, ValueError):
                idx_val = None

            offset_val: Optional[float]
            try:
                offset_val = float(sample.get("offset"))
            except (TypeError, ValueError):
                offset_val = None

            x_raw = sample.get("x")
            y_raw = sample.get("y")
            try:
                x_val = float(x_raw) if x_raw is not None else None
            except (TypeError, ValueError):
                x_val = None
            try:
                y_val = float(y_raw) if y_raw is not None else None
            except (TypeError, ValueError):
                y_val = None

            if (
                curv_val is not None
                and math.isfinite(curv_val)
                and x_val is not None
                and y_val is not None
            ):
                prepared_samples.append((x_val, y_val, curv_val))
                if path_key is not None or lane_key is not None:
                    lane_samples_xy.setdefault((path_key, lane_key), []).append(
                        (x_val, y_val, curv_val)
                    )

            if (
                path_key is None
                or lane_key is None
                or idx_val is None
                or curv_val is None
                or not math.isfinite(curv_val)
            ):
                continue

            idx_key = int(round(idx_val))
            bucket = curvature_by_index.setdefault((path_key, lane_key), {})
            bucket.setdefault(idx_key, []).append((offset_val, curv_val))

            if offset_val is not None and math.isfinite(offset_val):
                offset_key = round(float(offset_val), 6)
                offset_bucket = curvature_by_offset.setdefault((path_key, lane_key), {})
                offset_bucket.setdefault(offset_key, []).append(curv_val)

    cell_size = 1.5
    sample_grid: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}
    if prepared_samples:
        for x_val, y_val, curv_val in prepared_samples:
            cell_x = int(math.floor(x_val / cell_size))
            cell_y = int(math.floor(y_val / cell_size))
            sample_grid.setdefault((cell_x, cell_y), []).append((x_val, y_val, curv_val))

    max_radius_sq = (cell_size * 1.5) ** 2

    def _nearest_curvature(x_val: float, y_val: float) -> Optional[float]:
        if not prepared_samples:
            return None

        cell_x = int(math.floor(x_val / cell_size))
        cell_y = int(math.floor(y_val / cell_size))
        best_curv: Optional[float] = None
        best_dist_sq = max_radius_sq

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = sample_grid.get((cell_x + dx, cell_y + dy))
                if not bucket:
                    continue
                for sx, sy, curv in bucket:
                    dist_sq = (sx - x_val) ** 2 + (sy - y_val) ** 2
                    if dist_sq < best_dist_sq:
                        best_dist_sq = dist_sq
                        best_curv = curv

        if best_curv is None:
            for sx, sy, curv in prepared_samples:
                dist_sq = (sx - x_val) ** 2 + (sy - y_val) ** 2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_curv = curv

        return best_curv

    lane_locators: Dict[
        Tuple[Optional[str], Optional[str]], Optional[Callable[[float, float], Optional[float]]]
    ] = {}

    def _lane_locator(
        key: Tuple[Optional[str], Optional[str]]
    ) -> Optional[Callable[[float, float], Optional[float]]]:
        if key in lane_locators:
            return lane_locators[key]

        samples = lane_samples_xy.get(key)
        if not samples:
            lane_locators[key] = None
            return None

        grid: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}
        for sx, sy, curv in samples:
            cell_x = int(math.floor(sx / cell_size))
            cell_y = int(math.floor(sy / cell_size))
            grid.setdefault((cell_x, cell_y), []).append((sx, sy, curv))

        def _nearest(x_val: float, y_val: float) -> Optional[float]:
            best_curv: Optional[float] = None
            best_dist_sq = max_radius_sq

            cell_x = int(math.floor(x_val / cell_size))
            cell_y = int(math.floor(y_val / cell_size))

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bucket = grid.get((cell_x + dx, cell_y + dy))
                    if not bucket:
                        continue
                    for sx, sy, curv in bucket:
                        dist_sq = (sx - x_val) ** 2 + (sy - y_val) ** 2
                        if dist_sq < best_dist_sq:
                            best_dist_sq = dist_sq
                            best_curv = curv

            if best_curv is None:
                for sx, sy, curv in samples:
                    dist_sq = (sx - x_val) ** 2 + (sy - y_val) ** 2
                    if dist_sq < best_dist_sq:
                        best_dist_sq = dist_sq
                        best_curv = curv

            return best_curv

        lane_locators[key] = _nearest
        return _nearest

    for idx in range(len(line_geom_df)):
        row = line_geom_df.iloc[idx]

        line_id = _canonical_numeric(row[line_id_col], allow_negative=True)
        if line_id is None:
            continue

        lat_val = _to_float(row[lat_col])
        lon_val = _to_float(row[lon_col])
        if lat_val is None or lon_val is None:
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
                "shape_index": [],
                "has_true": False,
                "has_false": False,
                "has_flag": False,
                "path_token": None,
                "lane_token": None,
            },
        )

        if path_col is not None and entry.get("path_token") is None:
            entry["path_token"] = _normalise_key(row[path_col])

        if lane_col is not None and entry.get("lane_token") is None:
            entry["lane_token"] = _normalise_key(row[lane_col])

        z_val = _to_float(row[z_col]) if z_col else None
        if z_val is not None:
            if not math.isfinite(z_val) or abs(z_val) >= 1e4:
                z_val = None
        if z_val is None:
            existing = entry.get("z")
            if existing:
                z_val = existing[-1]
            else:
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
        entry["shape_index"].append(float(index))

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
        shape_indices = entry.get("shape_index", []) or []

        if not lat_vals or not lon_vals or not z_vals or not offsets_raw:
            continue

        try:
            x_vals, y_vals = latlon_to_local_xy(lat_vals, lon_vals, lat0, lon0)
        except Exception:  # pragma: no cover - defensive
            continue

        if len(x_vals) != len(z_vals) or len(x_vals) != len(offsets_raw):
            continue

        if len(shape_indices) != len(offsets_raw):
            if len(shape_indices) < len(offsets_raw):
                shape_indices = shape_indices + [None] * (len(offsets_raw) - len(shape_indices))
            else:
                shape_indices = shape_indices[: len(offsets_raw)]

        # The Japanese datasets reuse the same ``line_id`` across multiple
        # carriageway segments.  Each time the offset restarts from zero the
        # geometry would jump back to the beginning of the alignment, producing
        # spaghetti-like lane lines in the viewer.  Split the sequence whenever
        # the longitudinal coordinate decreases noticeably so every emitted
        # polyline remains strictly monotonic along ``s``.
        sequences: List[List[Tuple[float, float, float, float, Optional[float], Optional[float]]]] = []
        current: List[Tuple[float, float, float, float, Optional[float], Optional[float]]] = []
        last_s: Optional[float] = None
        reset_threshold = 1e-4  # tolerate sub-millimetre jitter while catching real resets

        entry_base_offset: Optional[float] = None
        for raw in offsets_raw:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if entry_base_offset is None or value < entry_base_offset:
                entry_base_offset = value

        for idx_point, (raw_offset, x_val, y_val, z_val) in enumerate(
            zip(offsets_raw, x_vals, y_vals, z_vals)
        ):
            try:
                offset_m = float(raw_offset)
            except (TypeError, ValueError):
                continue

            absolute_offset = offset_m

            if entry_base_offset is not None and math.isfinite(entry_base_offset):
                if (
                    global_base_offset is not None
                    and math.isfinite(global_base_offset)
                    and entry_base_offset > global_base_offset + 1e-6
                ):
                    offset_m -= global_base_offset
                else:
                    offset_m -= entry_base_offset
            elif global_base_offset is not None and math.isfinite(global_base_offset):
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
                prev_s, prev_x, prev_y, prev_z, _, _ = current[-1]
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

            shape_idx_val = None
            if idx_point < len(shape_indices):
                shape_idx_val = shape_indices[idx_point]
            current.append((s_float, x_float, y_float, z_float, shape_idx_val, absolute_offset))
            last_s = s_float

        if len(current) >= 2:
            sequences.append(current)

        geoms = lookup.setdefault(entry["line_id"], [])

        for seq in sequences:
            points = list(seq)
            signature = _geometry_signature([(p[0], p[1], p[2], p[3]) for p in points])
            if any(
                _geometry_signature(list(zip(g["s"], g["x"], g["y"], g["z"]))) == signature
                for g in geoms
            ):
                continue

            curvature_vals: List[Optional[float]] = [None] * len(points)
            path_token = entry.get("path_token")
            lane_token = entry.get("lane_token")
            lane_key = (path_token, lane_token)
            locator = _lane_locator(lane_key)

            direct_lookup: Optional[Dict[int, List[Tuple[Optional[float], float]]]] = None
            offset_lookup: Optional[Dict[float, List[float]]] = None
            if path_token is not None and lane_token is not None:
                direct_lookup = curvature_by_index.get(lane_key)
                offset_lookup = curvature_by_offset.get(lane_key)

            for idx_point, (
                _,
                x_coord,
                y_coord,
                _,
                shape_idx,
                absolute_offset,
            ) in enumerate(points):
                best_curv: Optional[float] = None

                if locator is not None:
                    best_curv = locator(x_coord, y_coord)

                if best_curv is None and direct_lookup and shape_idx is not None:
                    idx_key = int(round(shape_idx))
                    candidates = direct_lookup.get(idx_key, [])
                    if candidates:
                        if absolute_offset is not None and math.isfinite(absolute_offset):
                            finite = [
                                item
                                for item in candidates
                                if item[0] is not None and math.isfinite(item[0])
                            ]
                            if finite:
                                best_offset, best_curv_val = min(
                                    finite,
                                    key=lambda item: abs(item[0] - absolute_offset),
                                )
                                if abs(best_offset - absolute_offset) <= 0.5:
                                    best_curv = best_curv_val
                            else:
                                best_curv = candidates[0][1]
                        else:
                            best_curv = candidates[0][1]

                if (
                    best_curv is None
                    and offset_lookup
                    and absolute_offset is not None
                    and math.isfinite(absolute_offset)
                ):
                    offset_key = round(float(absolute_offset), 6)
                    candidates_off = offset_lookup.get(offset_key)
                    if not candidates_off and offset_lookup:
                        try:
                            best_key = min(
                                offset_lookup.keys(),
                                key=lambda key_val: abs(key_val - offset_key),
                            )
                            candidates_off = offset_lookup.get(best_key)
                        except ValueError:
                            candidates_off = None
                    if candidates_off:
                        best_curv = sum(candidates_off) / len(candidates_off)

                curvature_vals[idx_point] = best_curv

            if prepared_samples:
                for idx_point, value in enumerate(curvature_vals):
                    if value is not None and math.isfinite(value):
                        continue
                    _, x_coord, y_coord, _, _, _ = points[idx_point]
                    curvature_vals[idx_point] = _nearest_curvature(x_coord, y_coord)

            geom_entry: Dict[str, Any] = {
                "s": [p[0] for p in points],
                "x": [p[1] for p in points],
                "y": [p[2] for p in points],
                "z": [p[3] for p in points],
            }

            if curvature_vals and len(curvature_vals) == len(points):
                if any(val is not None for val in curvature_vals):
                    geom_entry["curvature"] = curvature_vals

            geoms.append(geom_entry)

    return lookup

