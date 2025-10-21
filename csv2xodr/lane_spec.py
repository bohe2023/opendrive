"""Helpers for building per-section lane specifications."""

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

from csv2xodr.mapping.core import mark_type_from_division_row

from csv2xodr.simpletable import DataFrame
from csv2xodr.topology.core import _canonical_numeric
from csv2xodr.normalize.core import latlon_to_local_xy


GEOMETRY_SIDE_CONFIDENCE_MIN = 0.5


def _clip_geometry_segment(geom: Dict[str, Any], s0: float, s1: float) -> Optional[Dict[str, List[float]]]:
    s_vals = geom.get("s") or []
    if not s_vals:
        return None

    x_vals = geom.get("x") or []
    y_vals = geom.get("y") or []
    z_vals = geom.get("z") or []
    curv_vals = geom.get("curvature") or []

    if len(s_vals) != len(x_vals) or len(s_vals) != len(y_vals) or len(s_vals) != len(z_vals):
        return None

    has_curvature = len(curv_vals) == len(s_vals)

    def _interpolate(a: float, b: float, ta: float, tb: float, target: float) -> float:
        if tb == ta:
            return a
        t = (target - ta) / (tb - ta)
        return a + t * (b - a)

    clipped: List[Tuple[float, float, float, float]] = []
    clipped_curv: List[Optional[float]] = []

    def _append(point: Tuple[float, float, float, float], curvature: Optional[float]) -> None:
        if not clipped or abs(clipped[-1][0] - point[0]) > 1e-6:
            clipped.append(point)
            if has_curvature:
                clipped_curv.append(curvature)

    for idx in range(len(s_vals) - 1):
        sa = s_vals[idx]
        sb = s_vals[idx + 1]
        xa, xb = x_vals[idx], x_vals[idx + 1]
        ya, yb = y_vals[idx], y_vals[idx + 1]
        za, zb = z_vals[idx], z_vals[idx + 1]
        ca = curv_vals[idx] if has_curvature else None
        cb = curv_vals[idx + 1] if has_curvature else None

        if sb <= s0 or sa >= s1:
            continue

        if sa < s0 <= sb:
            x_new = _interpolate(xa, xb, sa, sb, s0)
            y_new = _interpolate(ya, yb, sa, sb, s0)
            z_new = _interpolate(za, zb, sa, sb, s0)
            if has_curvature and ca is not None and cb is not None and sb != sa:
                t = (s0 - sa) / (sb - sa)
                c_new = ca + t * (cb - ca)
            else:
                c_new = ca if has_curvature else None
            _append((s0, x_new, y_new, z_new), c_new)

        if s0 <= sa <= s1:
            _append((sa, xa, ya, za), ca if has_curvature else None)

        if sa < s1 <= sb:
            x_new = _interpolate(xa, xb, sa, sb, s1)
            y_new = _interpolate(ya, yb, sa, sb, s1)
            z_new = _interpolate(za, zb, sa, sb, s1)
            if has_curvature and ca is not None and cb is not None and sb != sa:
                t = (s1 - sa) / (sb - sa)
                c_new = ca + t * (cb - ca)
            else:
                c_new = cb if has_curvature else None
            _append((s1, x_new, y_new, z_new), c_new)
        elif s0 <= sb <= s1:
            _append((sb, xb, yb, zb), cb if has_curvature else None)

    if not clipped:
        if len(s_vals) == 1 and s0 <= s_vals[0] <= s1:
            clipped.append((s_vals[0], x_vals[0], y_vals[0], z_vals[0]))
            if has_curvature:
                clipped_curv.append(curv_vals[0])

    if not clipped:
        return None

    result = {
        "s": [p[0] for p in clipped],
        "x": [p[1] for p in clipped],
        "y": [p[2] for p in clipped],
        "z": [p[3] for p in clipped],
    }

    if has_curvature and len(clipped_curv) == len(clipped):
        result["curvature"] = clipped_curv

    return result


def _lookup_line_segment(entry: Dict[str, Any], s0: float, s1: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, List[float]]]]:
    segments = entry.get("segments", []) if entry else []
    selected = None
    for seg in segments:
        if seg["s1"] <= s0:
            continue
        if seg["s0"] >= s1:
            continue
        selected = seg
        break
    if selected is None and segments:
        selected = segments[0]

    geom_segment = None
    geom_entries = entry.get("geometry", []) if entry else []
    if geom_entries:
        clip_start = s0
        clip_end = s1
        if selected is not None:
            clip_start = max(clip_start, selected["s0"])
            clip_end = min(clip_end, selected["s1"])
        if clip_end > clip_start:
            for geom in geom_entries:
                geom_s = geom.get("s") or []
                if not geom_s:
                    continue
                g0 = min(geom_s)
                g1 = max(geom_s)
                if g1 <= clip_start or g0 >= clip_end:
                    continue
                geom_segment = _clip_geometry_segment(geom, clip_start, clip_end)
                if geom_segment is not None:
                    break

    return selected, geom_segment


def _interpolate_centerline_pose(centerline: DataFrame, target_s: float) -> Tuple[float, float, float]:
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
            if span <= 0.0:
                return x_vals[idx], y_vals[idx], hdg_vals[idx]
            t = (target_s - s_prev) / span
            x = x_vals[idx - 1] + t * (x_vals[idx] - x_vals[idx - 1])
            y = y_vals[idx - 1] + t * (y_vals[idx] - y_vals[idx - 1])
            if abs(target_s - s_curr) <= 1e-6:
                hdg = hdg_vals[min(idx, len(hdg_vals) - 1)]
            else:
                hdg = hdg_vals[idx - 1]
            return x, y, hdg

    return x_vals[-1], y_vals[-1], hdg_vals[-1]


def _estimate_lane_side_from_geometry(
    lanes_geom_df: Optional[DataFrame],
    centerline: Optional[DataFrame],
    *,
    offset_mapper=None,
    geo_origin: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]:
    if (
        lanes_geom_df is None
        or len(lanes_geom_df) == 0
        or centerline is None
        or len(centerline) == 0
    ):
        return {}, {}, {}

    cols = list(lanes_geom_df.columns)

    def find_col(*keywords: str) -> Optional[str]:
        for col in cols:
            stripped = col.strip()
            if all(keyword in stripped for keyword in keywords):
                return col
        return None

    lane_id_col = find_col("Lane", "ID") or find_col("レーンID")
    lat_col = find_col("緯度") or find_col("Latitude")
    lon_col = find_col("経度") or find_col("Longitude")
    offset_col = find_col("Offset")

    if lane_id_col is None or lat_col is None or lon_col is None or offset_col is None:
        return {}, {}, {}

    try:
        lat_vals = [float(v) for v in lanes_geom_df[lat_col].astype(float).to_list()]
        lon_vals = [float(v) for v in lanes_geom_df[lon_col].astype(float).to_list()]
        offsets_cm = [float(v) for v in lanes_geom_df[offset_col].astype(float).to_list()]
    except Exception:
        return {}, {}, {}

    # ``Offset`` values in the raw CSV are absolute centimetre positions measured from
    # a global reference.  The centreline normalisation logic rebases them so that the
    # first valid sample maps to ``s = 0``.  Mirror that behaviour here to avoid feeding
    # extremely large numbers into the mapper (which would collapse everything to the
    # end of the alignment and cause lane sides to flip erratically).
    offsets_m: List[float] = [value / 100.0 for value in offsets_cm]
    base_offset_m: Optional[float] = None
    for value in offsets_m:
        if not math.isfinite(value):
            continue
        if base_offset_m is None or value < base_offset_m:
            base_offset_m = value
    if base_offset_m is not None:
        offsets_m = [value - base_offset_m if math.isfinite(value) else value for value in offsets_m]
    else:
        offsets_m = [0.0 for _ in offsets_m]

    lane_ids = [
        _canonical_numeric(value)
        for value in lanes_geom_df[lane_id_col].to_list()
    ]

    if not lane_ids or len(lane_ids) != len(offsets_m):
        return {}, {}, {}

    if geo_origin is not None:
        lat0, lon0 = geo_origin
    else:
        lat0 = lat_vals[0]
        lon0 = lon_vals[0]

    x_vals, y_vals = latlon_to_local_xy(lat_vals, lon_vals, lat0, lon0)

    side_samples: Dict[str, List[float]] = {}

    for lane_id, off_m, px, py in zip(lane_ids, offsets_m, x_vals, y_vals):
        if lane_id is None:
            continue
        try:
            s_val = float(off_m)
        except (TypeError, ValueError):
            continue
        if offset_mapper is not None:
            try:
                s_val = float(offset_mapper(s_val))
            except Exception:
                continue

        cx, cy, hdg = _interpolate_centerline_pose(centerline, s_val)
        dx = px - cx
        dy = py - cy
        left_x = -math.sin(hdg)
        left_y = math.cos(hdg)
        signed = dx * left_x + dy * left_y
        if not math.isfinite(signed):
            continue
        side_samples.setdefault(lane_id, []).append(signed)

    side_map: Dict[str, str] = {}
    strength_map: Dict[str, float] = {}
    bias_map: Dict[str, float] = {}

    lane_bias: Dict[str, float] = {}
    for lane_id, values in side_samples.items():
        if not values:
            continue
        lane_bias[lane_id] = sum(values) / len(values)

    global_shift = 0.0
    apply_shift = False
    if len(lane_bias) >= 2:
        values = list(lane_bias.values())
        min_val = min(values)
        max_val = max(values)
        if max_val - min_val > 1e-3:
            try:
                candidate = statistics.median(values)
            except statistics.StatisticsError:  # pragma: no cover - defensive
                candidate = 0.0
            if math.isfinite(candidate) and abs(candidate) > 0.05:
                global_shift = candidate
                apply_shift = True

    for lane_id, values in side_samples.items():
        if not values:
            continue
        shift = global_shift if apply_shift else 0.0
        avg = lane_bias.get(lane_id, 0.0) - shift
        if math.isfinite(avg):
            bias_map[lane_id] = avg
        if abs(avg) <= 0.05:
            continue
        side_map[lane_id] = "left" if avg > 0.0 else "right"
        strength_map[lane_id] = sum(abs(v - shift) for v in values) / len(values)

    return side_map, strength_map, bias_map


def _build_division_lookup(
    lane_div_df: Optional[DataFrame],
    line_geometry_lookup: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    offset_mapper=None,
) -> Dict[str, Dict[str, Any]]:
    if lane_div_df is None or len(lane_div_df) == 0:
        return {}

    cols = list(lane_div_df.columns)

    def find_col(*keywords: str) -> Optional[str]:
        for col in cols:
            stripped = col.strip()
            if all(keyword in stripped for keyword in keywords):
                return col
        return None

    start_col = find_col("Offset")
    end_col = find_col("End", "Offset")
    line_id_col = find_col("区画線ID") or find_col("ライン", "ID") or find_col("対象の区画線ID")
    if line_id_col is None:
        for col in cols:
            lowered = col.strip().lower()
            if "lane line id" in lowered and "数" not in lowered:
                line_id_col = col
                break
    start_w_col = find_col("始点側線幅")
    end_w_col = find_col("終点側線幅")
    width_code_col = None
    for col in cols:
        lowered = col.strip().lower()
        if "lane line width" in lowered:
            width_code_col = col
            break
    is_retrans_col = find_col("Is", "Retransmission")

    if line_id_col is None or start_col is None or end_col is None:
        return {}

    raw_records: List[Dict[str, Any]] = []
    for i in range(len(lane_div_df)):
        row = lane_div_df.iloc[i]
        try:
            start = float(row[start_col]) / 100.0
            end = float(row[end_col]) / 100.0
        except Exception:
            continue
        if end <= start:
            continue

        line_id_raw = row[line_id_col]
        line_id = _canonical_numeric(line_id_raw, allow_negative=True)
        if line_id is None:
            continue

        width_values: List[float] = []
        for col in (start_w_col, end_w_col):
            if not col:
                continue
            try:
                val = float(row[col]) / 100.0
                if val > 0:
                    width_values.append(val)
            except Exception:
                continue
        width = sum(width_values) / len(width_values) if width_values else None
        if width is None and width_code_col is not None:
            try:
                code = int(float(str(row[width_code_col]).strip()))
            except Exception:
                code = None
            if code is not None:
                width_lookup = {0: 0.10, 1: 0.15, 2: 0.20}
                mapped = width_lookup.get(code)
                if mapped is not None:
                    width = mapped

        is_retrans = False
        if is_retrans_col:
            is_retrans = str(row[is_retrans_col]).strip().lower() == "true"

        raw_records.append({
            "line_id": line_id,
            "start": start,
            "end": end,
            "row": row,
            "width": width,
            "is_retrans": is_retrans,
        })

    if not raw_records:
        return {}

    base_offset_m = min(record["start"] for record in raw_records)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in raw_records:
        adj_start = record["start"] - base_offset_m
        adj_end = record["end"] - base_offset_m

        mapped_start = adj_start
        mapped_end = adj_end
        if offset_mapper is not None:
            mapped_start = offset_mapper(mapped_start)
            mapped_end = offset_mapper(mapped_end)

        grouped.setdefault(record["line_id"], []).append(
            {
                "row": record["row"],
                "width": record["width"],
                "is_retrans": record["is_retrans"],
                "s0": mapped_start,
                "s1": mapped_end,
            }
        )

    lookup: Dict[str, Dict[str, Any]] = {}
    for line_id, segments in grouped.items():
        segments.sort(key=lambda item: (item["s0"], item["s1"]))
        cleaned: List[Dict[str, Any]] = []
        for data in segments:
            start = data["s0"]
            end = data["s1"]
            if cleaned:
                prev = cleaned[-1]
                if (
                    abs(prev["s0"] - start) < 1e-6
                    and abs(prev["s1"] - end) < 1e-6
                ):
                    prev_retrans = prev.get("_is_retrans", False)
                    new_retrans = data.get("is_retrans", False)
                    if prev_retrans == new_retrans or (prev_retrans and not new_retrans):
                        continue
                    if new_retrans and not prev_retrans:
                        mark_type = mark_type_from_division_row(data["row"])
                        cleaned[-1] = {
                            "s0": data["s0"],
                            "s1": data["s1"],
                            "type": mark_type,
                            "width": data["width"],
                            "_is_retrans": True,
                        }
                    continue
                if start < prev["s1"]:
                    start = max(prev["s1"], start)
                    if start >= end:
                        continue
                    data = data.copy()
                    data["s0"] = start

            mark_type = mark_type_from_division_row(data["row"])
            cleaned.append(
                {
                    "s0": data["s0"],
                    "s1": data["s1"],
                    "type": mark_type,
                    "width": data["width"],
                    "_is_retrans": data.get("is_retrans", False),
                }
            )

        if cleaned:
            for seg in cleaned:
                seg.pop("_is_retrans", None)
            lookup[line_id] = {
                "segments": cleaned,
                "geometry": (line_geometry_lookup or {}).get(line_id, []),
            }

    if line_geometry_lookup:
        for line_id, geoms in line_geometry_lookup.items():
            lookup.setdefault(line_id, {"segments": [], "geometry": geoms})

    return lookup



def build_lane_spec(
    sections: Iterable[Dict[str, Any]],
    lane_topo: Optional[Dict[str, Any]],
    defaults: Dict[str, Any],
    lane_div_df,
    *,
    line_geometry_lookup: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    offset_mapper=None,
    lanes_geometry_df: Optional[DataFrame] = None,
    centerline: Optional[DataFrame] = None,
    geo_origin: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """Return metadata for each lane section used by the writer."""

    sections_list = list(sections)
    lane_info = (lane_topo or {}).get("lanes") or {}
    raw_lane_groups = (lane_topo or {}).get("groups") or {}
    lane_count = (lane_topo or {}).get("lane_count") or 0

    if not lane_info:
        # fallback to the previous heuristic
        lanes_guess = [1, -1]
        total = len(sections_list)
        out: List[Dict[str, Any]] = []
        roadmark_type = "solid"
        if lane_div_df is not None and len(lane_div_df) > 0:
            roadmark_type = mark_type_from_division_row(lane_div_df.iloc[0])
        for idx, sec in enumerate(sections_list):
            out.append(
                {
                    "s0": sec["s0"],
                    "s1": sec["s1"],
                    "left": [
                        {
                            "id": 1,
                            "width": defaults.get("lane_width_m", 3.5),
                            "roadMark": {
                                "type": roadmark_type,
                                "width": 0.12,
                                "laneChange": "both" if roadmark_type != "solid" else "none",
                            },
                            "successors": [1] if idx < total - 1 else [],
                            "predecessors": [1] if idx > 0 else [],
                        }
                    ],
                    "right": [
                        {
                            "id": -1,
                            "width": defaults.get("lane_width_m", 3.5),
                            "roadMark": {
                                "type": roadmark_type,
                                "width": 0.12,
                                "laneChange": "both" if roadmark_type != "solid" else "none",
                            },
                            "successors": [-1] if idx < total - 1 else [],
                            "predecessors": [-1] if idx > 0 else [],
                        }
                    ],
                }
            )
        return out

    def _split_lane_groups(
        groups: Dict[str, List[str]],
        lane_data: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
        """Split lane groups that mix positive/negative lane numbers."""

        expanded: Dict[str, List[str]] = {}
        parent_map: Dict[str, str] = {}
        derived_hints: Dict[str, str] = {}

        for base_id, uids in groups.items():
            positive: List[str] = []
            negative: List[str] = []
            neutral: List[str] = []

            for uid in uids:
                info = lane_data.get(uid) or {}
                lane_no = info.get("lane_no")
                if isinstance(lane_no, (int, float)):
                    if lane_no > 0:
                        positive.append(uid)
                    elif lane_no < 0:
                        negative.append(uid)
                    else:
                        neutral.append(uid)
                else:
                    neutral.append(uid)

            buckets: List[Tuple[str, List[str]]] = []
            if positive:
                buckets.append(("pos", positive))
            if negative:
                buckets.append(("neg", negative))
            if neutral and (positive or negative):
                buckets.append(("zero", neutral))

            if len(buckets) <= 1:
                expanded[base_id] = list(uids)
                parent_map[base_id] = base_id
                if positive and not negative:
                    derived_hints[base_id] = "left"
                elif negative and not positive:
                    derived_hints[base_id] = "right"
                continue

            for suffix, items in buckets:
                alias = f"{base_id}::{suffix}"
                expanded[alias] = items
                parent_map[alias] = base_id
                if suffix == "pos":
                    derived_hints[alias] = "left"
                elif suffix == "neg":
                    derived_hints[alias] = "right"
                else:
                    derived_hints.setdefault(alias, "center")

        return expanded, parent_map, derived_hints

    lane_groups, group_parent_map, derived_side_hints = _split_lane_groups(raw_lane_groups, lane_info)

    division_lookup = _build_division_lookup(
        lane_div_df, line_geometry_lookup=line_geometry_lookup, offset_mapper=offset_mapper
    )

    (
        raw_geometry_side_hint,
        raw_geometry_strength,
        raw_geometry_bias,
    ) = _estimate_lane_side_from_geometry(
        lanes_geometry_df,
        centerline,
        offset_mapper=offset_mapper,
        geo_origin=geo_origin,
    )

    parent_lane_signs: Dict[str, Tuple[bool, bool]] = {}
    for base_id, members in raw_lane_groups.items():
        has_positive = False
        has_negative = False
        for uid in members:
            lane_no = lane_info.get(uid, {}).get("lane_no")
            if isinstance(lane_no, (int, float)):
                if lane_no > 0:
                    has_positive = True
                elif lane_no < 0:
                    has_negative = True
            if has_positive and has_negative:
                break
        parent_lane_signs[base_id] = (has_positive, has_negative)

    geometry_side_hint: Dict[str, str] = {}
    geometry_hint_strength: Dict[str, float] = {}
    geometry_bias: Dict[str, float] = {
        key: float(value)
        for key, value in (raw_geometry_bias or {}).items()
        if value is not None and math.isfinite(value)
    }
    for alias, parent in group_parent_map.items():
        hint = derived_side_hints.get(alias)
        parent_hint = raw_geometry_side_hint.get(parent)
        parent_strength = raw_geometry_strength.get(parent, 0.0)
        if parent_hint in {"left", "right"}:
            signs = parent_lane_signs.get(parent)
            if signs is not None:
                has_pos, has_neg = signs
                if has_pos and has_neg:
                    # 几何估计只适用于车道组位于参考线单侧的情况；
                    # 当同一组同时包含正负 lane_no 时说明其跨越中心线，
                    # 继续沿用几何提示会把所有车道推到同一侧，导致
                    # 导出的道路在查看器中出现交错的白线。此时宁可
                    # 完全依赖拓扑信息，也不要使用这样的提示。
                    parent_hint = None
                    parent_strength = 0.0
        if hint in {"left", "right"}:
            if parent_hint in {"left", "right"} and parent_hint != hint:
                if alias == parent:
                    # 当几何信息与车道编号推断相矛盾时，检查该组是否只包含
                    # 单侧编号（全部为正或全部为负）。如果是这样，则更信任
                    # 编号推断，以免把整组车道硬塞到错误的一侧，造成输出
                    # 出现交错的白线。
                    strength = raw_geometry_strength.get(parent, 0.0)
                    if strength >= GEOMETRY_SIDE_CONFIDENCE_MIN:
                        geometry_side_hint[alias] = parent_hint
                        geometry_hint_strength[alias] = strength
                    else:
                        signs = parent_lane_signs.get(parent)
                        if signs is not None:
                            has_pos, has_neg = signs
                        else:
                            has_pos = has_neg = False
                        if has_pos and has_neg:
                            geometry_side_hint[alias] = parent_hint
                            geometry_hint_strength[alias] = strength
                        else:
                            geometry_side_hint[alias] = hint
                else:
                    # For split groups ("::pos"/"::neg") the derived hint
                    # still carries meaningful information about how the
                    # sub-group relates to the reference line.
                    geometry_side_hint[alias] = hint
            else:
                geometry_side_hint[alias] = hint
            continue

        if parent_hint in {"left", "right"}:
            geometry_side_hint[alias] = parent_hint
            if parent_strength >= GEOMETRY_SIDE_CONFIDENCE_MIN:
                geometry_hint_strength[alias] = parent_strength


    def _segment_spans(info: Dict[str, Any]) -> List[Tuple[float, float]]:
        spans: List[Tuple[float, float]] = []
        for seg in info.get("segments", []):
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            spans.append((start, end))
        spans.sort(key=lambda item: item[0])
        return spans

    def _mergeable(existing: List[Tuple[float, float]], new_spans: List[Tuple[float, float]], *, tolerance: float = 1e-3) -> bool:
        if not existing:
            return True
        for s0, s1 in new_spans:
            for e0, e1 in existing:
                if s1 <= e0 + tolerance or s0 >= e1 - tolerance:
                    continue
                return False
        return True

    lane_no_by_base: Dict[str, Optional[int]] = {}
    base_start: Dict[str, float] = {}
    lane_spans: Dict[str, List[Tuple[float, float]]] = {}

    for base_id, uids in lane_groups.items():
        lane_numbers: List[int] = []
        first_start: Optional[float] = None
        for uid in uids:
            info = lane_info.get(uid) or {}
            lane_no = info.get("lane_no")
            if isinstance(lane_no, (int, float)):
                lane_numbers.append(int(lane_no))
            spans = _segment_spans(info)
            if spans:
                lane_spans[uid] = spans
                if first_start is None or spans[0][0] < first_start:
                    first_start = spans[0][0]
        lane_no_by_base[base_id] = min(lane_numbers) if lane_numbers else None
        base_start[base_id] = first_start or 0.0

    def _base_sort_key(base_id: str) -> Tuple[int, float, str]:
        lane_number = lane_no_by_base.get(base_id)
        return (lane_number if lane_number is not None else 0, base_start.get(base_id, 0.0), base_id)

    base_ids = sorted(lane_groups.keys(), key=_base_sort_key)
    lanes_per_base = {base: len(lane_groups[base]) for base in base_ids}

    for base_id in base_ids:
        lane_number = lane_no_by_base.get(base_id)
        if lane_number is None or lane_number == 0:
            continue
        expected_side = "left" if lane_number > 0 else "right"
        current = geometry_side_hint.get(base_id)
        if current is None:
            geometry_side_hint[base_id] = expected_side
        elif current != expected_side:
            strength = geometry_hint_strength.get(base_id, 0.0)
            parent = group_parent_map.get(base_id)
            if strength < GEOMETRY_SIDE_CONFIDENCE_MIN and parent is not None:
                strength = geometry_hint_strength.get(parent, strength)

            signs = parent_lane_signs.get(base_id)
            if signs is None and parent is not None:
                signs = parent_lane_signs.get(parent)
            has_pos = has_neg = False
            if signs is not None:
                has_pos, has_neg = signs

            if not (has_pos and has_neg):
                if strength is not None and strength >= GEOMETRY_SIDE_CONFIDENCE_MIN:
                    continue
                geometry_side_hint[base_id] = expected_side
                continue

            if strength is not None and strength >= GEOMETRY_SIDE_CONFIDENCE_MIN:
                continue

            geometry_side_hint[base_id] = expected_side

    def _bases_with_sign(sign: int) -> List[str]:
        selected: List[str] = []
        for base in base_ids:
            for uid in lane_groups.get(base, []):
                lane_no = lane_info.get(uid, {}).get("lane_no")
                if lane_no is None:
                    continue
                if sign > 0 and lane_no > 0:
                    selected.append(base)
                    break
                if sign < 0 and lane_no < 0:
                    selected.append(base)
                    break
        return selected

    positive_bases = _bases_with_sign(1)
    negative_bases = _bases_with_sign(-1)
    ordered_lane_numbers = [
        lane_no
        for lane_no in sorted(
            {val for val in lane_no_by_base.values() if val is not None}
        )
    ]

    if lane_count and len(ordered_lane_numbers) > lane_count:
        ordered_lane_numbers = ordered_lane_numbers[:lane_count]

    default_lane_side = defaults.get("default_lane_side", "left")
    if default_lane_side not in {"left", "right"}:
        default_lane_side = "left"
    default_lane_side_is_right = default_lane_side == "right"

    has_geometry_right_hint = any(side == "right" for side in geometry_side_hint.values())
    only_positive_without_initial_right_evidence = (
        not negative_bases and not has_geometry_right_hint and bool(positive_bases)
    )

    force_single_side_left = False

    if only_positive_without_initial_right_evidence:
        # 没有任何右侧提示且全部车道编号为正，说明输入数据没有明确区分两侧。
        # 之前的逻辑会尝试根据 lane_count 等参数重新划分左右两侧，反而会把
        # 一部分车道强制分配到右侧。这里改为保持默认，让后续流程按照既有的
        # 提示或车道编号推导结果继续处理。
        for base in base_ids:
            if lane_no_by_base.get(base) is None:
                continue
            geometry_side_hint.setdefault(base, "left")
        force_single_side_left = True

    def _ordered_subset(candidates: Iterable[str]) -> List[str]:
        seen: List[str] = []
        for base in base_ids:
            if base in candidates and base not in seen:
                seen.append(base)
        return seen

    hinted_left = _ordered_subset(
        [base for base, side in geometry_side_hint.items() if side == "left"]
    )
    hinted_right = _ordered_subset(
        [base for base, side in geometry_side_hint.items() if side == "right"]
    )

    remaining_bases = [
        base
        for base in base_ids
        if base not in hinted_left and base not in hinted_right
    ]

    derived_left: List[str] = []
    derived_right: List[str] = []

    def _single_side_positive_only() -> bool:
        return not negative_bases and not hinted_right and not derived_right

    if remaining_bases:
        single_side_without_right_hint = (
            not negative_bases
            and not hinted_right
            and not has_geometry_right_hint
        )

        if single_side_without_right_hint:
            derived_left = list(remaining_bases)
            derived_right = []
            force_single_side_left = True
        elif not negative_bases and not has_geometry_right_hint and not hinted_right:
            derived_left = list(remaining_bases)
            derived_right = []
        elif positive_bases and negative_bases:
            derived_left = _ordered_subset(
                [base for base in positive_bases if base in remaining_bases]
            )
            derived_right = _ordered_subset(
                [base for base in negative_bases if base in remaining_bases]
            )
        else:
            has_right_evidence = bool(
                negative_bases or hinted_right or has_geometry_right_hint
            )
            has_left_evidence = bool(positive_bases or hinted_left)

            if not has_right_evidence:
                derived_left = list(remaining_bases)
                derived_right = []
            elif not has_left_evidence:
                derived_right = list(remaining_bases)
                derived_left = []
            elif _single_side_positive_only():
                if default_lane_side_is_right:
                    derived_left = []
                    derived_right = list(remaining_bases)
                else:
                    derived_left = list(remaining_bases)
                    derived_right = []
            else:
                # 根据已有的车道编号符号优先推断左右侧，剩余的按照默认侧放置。
                derived_left = _ordered_subset(
                    [
                        base
                        for base in positive_bases
                        if base in remaining_bases
                    ]
                )
                derived_right = _ordered_subset(
                    [
                        base
                        for base in negative_bases
                        if base in remaining_bases
                    ]
                )

                unresolved = [
                    base
                    for base in remaining_bases
                    if base not in derived_left and base not in derived_right
                ]

                if unresolved:
                    preferred_side = (
                        "right" if default_lane_side_is_right else "left"
                    )
                    if preferred_side == "left" or not has_right_evidence:
                        derived_left.extend(unresolved)
                    else:
                        derived_right.extend(unresolved)

    # `derived_right` 仅代表算法在缺乏直接信息时的暂时推断。如果没有来自
    # 编号或几何的右侧证据，就不应该因为之前的推断而放弃“全部保持在同一侧”
    # 的兜底策略，否则会把单侧车道硬性拆成左右两组。
    final_no_right_evidence = (
        not negative_bases
        and not hinted_right
        and not has_geometry_right_hint
    )

    if final_no_right_evidence:
        force_single_side_left = True

    only_positive_without_right_evidence = bool(positive_bases) and final_no_right_evidence

    if remaining_bases and _single_side_positive_only() and not force_single_side_left:
        if default_lane_side_is_right:
            derived_left = []
            derived_right = list(remaining_bases)
        else:
            derived_left = list(remaining_bases)
            derived_right = []

    left_bases = hinted_left + [
        base for base in derived_left if base not in hinted_left and base not in hinted_right
    ]
    right_bases = hinted_right + [
        base for base in derived_right if base not in hinted_left and base not in hinted_right
    ]

    force_all_default_side = False
    skip_unassigned_assignment = False

    if force_single_side_left:
        left_bases = [base for base in base_ids]
        right_bases = []
        skip_unassigned_assignment = True
    else:
        force_all_default_side = final_no_right_evidence
        no_right_side_assignments = final_no_right_evidence

        if force_all_default_side:
            if default_lane_side_is_right:
                left_bases = []
                right_bases = [base for base in base_ids]
            else:
                left_bases = [base for base in base_ids]
                right_bases = []
            skip_unassigned_assignment = True

        if not hinted_left and not hinted_right and not force_all_default_side:
            has_right_evidence = bool(
                negative_bases
                or hinted_right
                or derived_right
                or has_geometry_right_hint
            )
            if (
                not has_right_evidence
                or only_positive_without_right_evidence
                or no_right_side_assignments
            ):
                left_bases = [base for base in base_ids]
                right_bases = []
                skip_unassigned_assignment = True
            else:
                if not left_bases and base_ids:
                    left_bases = base_ids[:1]
                    right_bases = [base for base in base_ids if base not in left_bases]
                elif not right_bases and base_ids:
                    right_bases = [base for base in base_ids if base not in left_bases]

    left_base_set = set(left_bases)
    right_base_set = set(right_bases)

    unassigned = [
        base for base in base_ids if base not in left_base_set and base not in right_base_set
    ]
    if (
        not skip_unassigned_assignment
        and not force_all_default_side
        and not _single_side_positive_only()
    ):
        for base in unassigned:
            neighbour_side: Optional[str] = None
            for uid in lane_groups.get(base, []):
                info = lane_info.get(uid) or {}
                segments = info.get("segments", [])
                for seg in segments:
                    left_neighbour = seg.get("left_neighbor")
                    right_neighbour = seg.get("right_neighbor")
                    if left_neighbour in left_base_set or right_neighbour in left_base_set:
                        neighbour_side = "left"
                        break
                    if left_neighbour in right_base_set or right_neighbour in right_base_set:
                        neighbour_side = "right"
                        break
                if neighbour_side:
                    break

            if neighbour_side == "left":
                left_bases.append(base)
                left_base_set.add(base)
            elif neighbour_side == "right":
                right_bases.append(base)
                right_base_set.add(base)
            elif len(left_bases) <= len(right_bases):
                left_bases.append(base)
                left_base_set.add(base)
            else:
                right_bases.append(base)
                right_base_set.add(base)

    if not left_bases and not right_bases and base_ids:
        if not negative_bases and not hinted_right and not derived_right:
            left_bases = [base for base in base_ids]
            right_bases = []
        else:
            left_bases = base_ids[:1]
            right_bases = [base for base in base_ids if base not in left_bases]
        left_base_set = set(left_bases)
        right_base_set = set(right_bases)

    lane_id_map: Dict[str, int] = {}
    lane_side_map: Dict[str, str] = {}

    assigned_spans_left: Dict[int, List[Tuple[float, float]]] = {}
    assigned_spans_right: Dict[int, List[Tuple[float, float]]] = {}

    def _reuse_candidate(
        uid: str,
        *,
        side: str,
        assigned_spans: Dict[int, List[Tuple[float, float]]],
    ) -> Optional[int]:
        info = lane_info.get(uid) or {}
        spans = lane_spans.get(uid, [])
        if not spans:
            return None

        candidates: List[int] = []
        for seg in info.get("segments", []):
            for key in ("predecessors", "successors"):
                for target in seg.get(key, []):
                    candidate = lane_id_map.get(target)
                    if candidate is None:
                        continue
                    if side == "left" and candidate <= 0:
                        continue
                    if side == "right" and candidate >= 0:
                        continue
                    if candidate not in candidates:
                        candidates.append(candidate)

        if not candidates:
            return None

        for candidate in sorted(candidates, key=lambda val: (abs(val), val)):
            existing = assigned_spans.get(candidate, [])
            if _mergeable(existing, spans):
                return candidate
        return None

    left_id_by_lane_no: Dict[int, int] = {}
    current_id = 1
    for base in left_bases:
        ordered = reversed(sorted(lane_groups.get(base, []), key=lambda x: lane_info[x]["lane_no"]))
        for uid in ordered:
            info = lane_info[uid]
            lane_no = info["lane_no"]
            key = int(lane_no)
            assigned = _reuse_candidate(uid, side="left", assigned_spans=assigned_spans_left)
            if assigned is None:
                assigned = left_id_by_lane_no.get(key)
            if assigned is None:
                assigned = current_id
                left_id_by_lane_no[key] = assigned
                current_id += 1
            lane_id_map[uid] = assigned
            lane_side_map[uid] = "left"
            spans = lane_spans.get(uid, [])
            if spans:
                assigned_spans_left.setdefault(assigned, []).extend(spans)

    right_id_by_lane_no: Dict[int, int] = {}
    current_id = -1
    for base in right_bases:
        ordered = reversed(sorted(lane_groups.get(base, []), key=lambda x: lane_info[x]["lane_no"]))
        for uid in ordered:
            info = lane_info[uid]
            lane_no = info["lane_no"]
            key = int(lane_no)
            assigned = _reuse_candidate(uid, side="right", assigned_spans=assigned_spans_right)
            if assigned is None:
                assigned = right_id_by_lane_no.get(key)
            if assigned is None:
                assigned = current_id
                right_id_by_lane_no[key] = assigned
                current_id -= 1
            lane_id_map[uid] = assigned
            lane_side_map[uid] = "right"
            spans = lane_spans.get(uid, [])
            if spans:
                assigned_spans_right.setdefault(assigned, []).extend(spans)

    lane_section_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    lane_section_indices: Dict[str, List[int]] = {uid: [] for uid in lane_id_map}
    for idx, sec in enumerate(sections_list):
        s0 = sec["s0"]
        s1 = sec["s1"]
        for uid, info in lane_info.items():
            if uid not in lane_id_map:
                continue
            segment = None
            for seg in info["segments"]:
                if seg["end"] <= s0 or seg["start"] >= s1:
                    continue
                segment = seg
                break
            if segment is None:
                continue
            lane_section_map[(uid, idx)] = segment
            lane_section_indices[uid].append(idx)

    lane_section_pos: Dict[Tuple[str, int], int] = {}
    for uid, indices in lane_section_indices.items():
        for order, sec_idx in enumerate(indices):
            lane_section_pos[(uid, sec_idx)] = order

    lane_segment_ranges: Dict[str, List[Tuple[float, float]]] = {}
    for uid, info in lane_info.items():
        segments = []
        for seg in info.get("segments", []):
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            segments.append((start, end))
        if segments:
            lane_segment_ranges[uid] = segments

    out: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections_list):
        section_left: List[Dict[str, Any]] = []
        section_right: List[Dict[str, Any]] = []
        s0 = sec["s0"]
        s1 = sec["s1"]

        for uid, lane_id in lane_id_map.items():
            segment = lane_section_map.get((uid, idx))
            if segment is None:
                continue

            side = lane_side_map[uid]
            lane_no = lane_info[uid]["lane_no"]
            width = segment.get("width") or defaults.get("lane_width_m", 3.5)

            if isinstance(width, str):
                try:
                    width = float(width)
                except Exception:
                    width = defaults.get("lane_width_m", 3.5)

            lane_indices = lane_section_indices[uid]
            order = lane_section_pos[(uid, idx)]
            prev_index = lane_indices[order - 1] if order > 0 else None
            next_index = lane_indices[order + 1] if order < len(lane_indices) - 1 else None

            current_start = float(segment.get("start", s0))
            current_end = float(segment.get("end", s1))

            predecessors: List[int] = []
            if order > 0 and prev_index is not None and idx - prev_index == 1:
                predecessors.append(lane_id)
            else:
                for target in segment.get("predecessors", []):
                    mapped = lane_id_map.get(target)
                    if mapped is None:
                        continue

                    ranges = lane_segment_ranges.get(target, [])
                    for start, end in ranges:
                        if end <= current_start + 1e-3:
                            if mapped not in predecessors:
                                predecessors.append(mapped)
                            break

            successors: List[int] = []
            if (
                next_index is not None
                and order < len(lane_indices) - 1
                and next_index - idx == 1
            ):
                successors.append(lane_id)
            else:
                for target in segment.get("successors", []):
                    mapped = lane_id_map.get(target)
                    if mapped is None:
                        continue

                    ranges = lane_segment_ranges.get(target, [])
                    for start, end in ranges:
                        if start >= current_end - 1e-3:
                            if mapped not in successors:
                                successors.append(mapped)
                            break

            pos_key = 2 if side == "left" else 1
            alt_key = 1 if side == "left" else 2

            line_positions = segment.get("line_positions", {}) or {}

            def _iter_ids(*keys):
                for key in keys:
                    value = line_positions.get(key)
                    if isinstance(value, (list, tuple)):
                        for item in value:
                            if item:
                                yield item
                    elif value:
                        yield value

            line_id = None
            candidates = list(_iter_ids(pos_key, alt_key))
            for candidate in candidates:
                if division_lookup.get(candidate):
                    line_id = candidate
                    break
            if line_id is None and candidates:
                line_id = candidates[0]

            mark = None
            if line_id:
                segment_entry = division_lookup.get(line_id)
                mark_segment, geom_segment = _lookup_line_segment(segment_entry, s0, s1)
                if mark_segment or geom_segment:
                    mark_width = (mark_segment.get("width") if mark_segment else None) or 0.12
                    mark_type = mark_segment.get("type") if mark_segment else None
                    lane_change = "both"
                    if mark_type == "solid":
                        lane_change = "none"
                    mark = {
                        "type": mark_type or "solid",
                        "width": mark_width,
                        "laneChange": lane_change,
                    }
                    if geom_segment:
                        mark["geometry"] = geom_segment

            if mark is None:
                lane_change = "both"
                mark = {"type": "solid", "width": 0.12, "laneChange": lane_change}

            lane_entry = {
                "uid": uid,
                "id": lane_id,
                "lane_no": lane_no,
                "width": width,
                "roadMark": mark,
                "predecessors": predecessors,
                "successors": successors,
                "type": lane_info[uid].get("type", "driving"),
            }

            if side == "left":
                section_left.append(lane_entry)
            else:
                section_right.append(lane_entry)

        section_left.sort(key=lambda item: item["id"])
        section_right.sort(key=lambda item: item["id"], reverse=True)

        lane_offset = _compute_lane_offset(section_left, section_right, lane_info, geometry_bias)

        section_entry = {"s0": s0, "s1": s1, "left": section_left, "right": section_right}
        if lane_offset is not None:
            section_entry["laneOffset"] = lane_offset

        out.append(section_entry)

    return out


def _compute_lane_offset(
    section_left: List[Dict[str, Any]],
    section_right: List[Dict[str, Any]],
    lane_info: Dict[str, Dict[str, Any]],
    geometry_bias: Dict[str, float],
) -> Optional[float]:
    """Estimate the lateral shift that recentres the lane stack."""

    if not geometry_bias:
        return None

    def _lane_center_bias(lane: Dict[str, Any]) -> Optional[float]:
        uid = lane.get("uid")
        info = lane_info.get(uid, {})
        base_id = info.get("base_id")
        if base_id is None:
            return None
        bias = geometry_bias.get(base_id)
        if bias is None or not math.isfinite(bias):
            return None
        return float(bias)

    center_biases: List[float] = []
    all_biases: List[float] = []
    has_left_bias = False
    has_right_bias = False

    for lane in section_left + section_right:
        lane_no = lane.get("lane_no")
        if lane_no is None:
            lane_no = lane_info.get(lane.get("uid"), {}).get("lane_no")
        if lane_no is None:
            continue
        try:
            lane_no_val = float(lane_no)
        except (TypeError, ValueError):
            continue
        bias = _lane_center_bias(lane)
        if bias is None:
            continue

        all_biases.append(bias)
        if bias > 0:
            has_left_bias = True
        elif bias < 0:
            has_right_bias = True

        if abs(lane_no_val) <= 0.5:
            center_biases.append(bias)

    if center_biases:
        # 中央车道已经直接采用 CSV 中的几何作为参考线，因此一旦识别到
        # lane_no ≈ 0 的车道，就不再尝试平移 lane stack。否则会把导出的
        # 车道整体挪离实测白线，引发用户报告的错位现象。
        return None

    if all_biases and has_left_bias and has_right_bias:
        candidate = min(all_biases, key=lambda value: abs(value))
        if math.isfinite(candidate) and abs(candidate) > 1e-3:
            return -candidate

    left_outer: List[float] = []
    right_outer: List[float] = []
    left_inner: List[float] = []
    right_inner: List[float] = []

    def _append(target: List[float], value: float) -> None:
        if math.isfinite(value):
            target.append(value)

    for lane in section_left:
        info = lane_info.get(lane.get("uid"), {})
        base_id = info.get("base_id")
        bias = geometry_bias.get(base_id)
        if bias is None:
            continue
        try:
            width_val = float(lane.get("width", 0.0))
        except (TypeError, ValueError):
            continue
        half = width_val * 0.5
        _append(left_outer, bias + half)
        _append(left_inner, bias - half)

    for lane in section_right:
        info = lane_info.get(lane.get("uid"), {})
        base_id = info.get("base_id")
        bias = geometry_bias.get(base_id)
        if bias is None:
            continue
        try:
            width_val = float(lane.get("width", 0.0))
        except (TypeError, ValueError):
            continue
        half = width_val * 0.5
        _append(right_outer, bias - half)
        _append(right_inner, bias + half)

    def _closest(values: List[float], *, prefer_positive: Optional[bool]) -> Optional[float]:
        if not values:
            return None
        filtered: List[float] = []
        if prefer_positive is True:
            filtered = [val for val in values if val >= 0.0]
        elif prefer_positive is False:
            filtered = [val for val in values if val <= 0.0]
        if filtered:
            values = filtered
        return min(values, key=lambda val: abs(val))

    left_edge = _closest(left_inner, prefer_positive=True)
    right_edge = _closest(right_inner, prefer_positive=False)

    center_bias: Optional[float] = None
    if left_edge is not None and right_edge is not None:
        center_bias = 0.5 * (left_edge + right_edge)
    elif left_outer and right_outer:
        center_bias = 0.5 * (max(left_outer) + min(right_outer))
    elif left_inner and left_outer:
        inner = _closest(left_inner, prefer_positive=True)
        if inner is not None:
            center_bias = 0.5 * (inner + max(left_outer))
    elif right_inner and right_outer:
        inner = _closest(right_inner, prefer_positive=False)
        if inner is not None:
            center_bias = 0.5 * (inner + min(right_outer))

    if center_bias is None:
        return None

    if not math.isfinite(center_bias) or abs(center_bias) <= 1e-3:
        return None

    return -center_bias


def apply_shoulder_profile(
    lane_sections: List[Dict[str, Any]],
    shoulder_segments: List[Dict[str, float]],
    *,
    defaults: Optional[Dict[str, Any]] = None,
) -> None:
    if not lane_sections:
        return

    defaults = defaults or {}
    default_width = float(defaults.get("shoulder_width_m", 0.0) or 0.0)

    def _average_width(s0: float, s1: float, side: str) -> Optional[float]:
        total = 0.0
        total_length = 0.0
        for seg in shoulder_segments:
            start = max(s0, seg["s0"])
            end = min(s1, seg["s1"])
            length = end - start
            if length <= 0:
                continue
            width = float(seg.get(side, 0.0))
            total += width * length
            total_length += length
        if total_length > 0:
            return total / total_length
        return None

    left_prev: Optional[Dict[str, Any]] = None
    right_prev: Optional[Dict[str, Any]] = None

    def _existing_lane_ids(side: str) -> List[int]:
        ids: List[int] = []
        for section in lane_sections:
            for lane in section.get(side, []):
                try:
                    lane_id = int(lane.get("id"))
                except (TypeError, ValueError):
                    continue
                ids.append(lane_id)
        return ids

    left_ids = _existing_lane_ids("left")
    right_ids = _existing_lane_ids("right")

    if left_ids:
        LEFT_ID = max(left_ids) + 1
        if LEFT_ID <= 0:
            LEFT_ID = 1
    else:
        LEFT_ID = 1

    if right_ids:
        RIGHT_ID = min(right_ids) - 1
        if RIGHT_ID >= 0:
            RIGHT_ID = -1
    else:
        RIGHT_ID = -1

    for section in lane_sections:
        s0 = section.get("s0", 0.0)
        s1 = section.get("s1", s0)

        left_width = _average_width(s0, s1, "left")
        right_width = _average_width(s0, s1, "right")

        if left_width is None:
            left_width = default_width
        if right_width is None:
            right_width = default_width

        if left_width and left_width > 0:
            entry = {
                "id": LEFT_ID,
                "type": "shoulder",
                "width": float(left_width),
                "roadMark": None,
                "predecessors": [LEFT_ID] if left_prev is not None else [],
                "successors": [],
            }
            section.setdefault("left", []).append(entry)
            if left_prev is not None:
                left_prev["successors"] = [LEFT_ID]
            left_prev = entry
        else:
            left_prev = None

        if right_width and right_width > 0:
            entry = {
                "id": RIGHT_ID,
                "type": "shoulder",
                "width": float(right_width),
                "roadMark": None,
                "predecessors": [RIGHT_ID] if right_prev is not None else [],
                "successors": [],
            }
            section.setdefault("right", []).append(entry)
            if right_prev is not None:
                right_prev["successors"] = [RIGHT_ID]
            right_prev = entry
        else:
            right_prev = None

        section["left"].sort(key=lambda item: item["id"])
        section["right"].sort(key=lambda item: item["id"], reverse=True)


def normalize_lane_ids(lane_sections: List[Dict[str, Any]]) -> None:
    """Renumber lane IDs so that they are compact and sequential."""

    if not lane_sections:
        return

    occurrence: Dict[Tuple[int, int], int] = {}

    # First pass: remember the original identifiers and assign new IDs based on
    # the positional order within each section.
    for sec_idx, section in enumerate(lane_sections):
        for side in ("left", "right"):
            lanes = section.get(side, []) or []
            sign = 1 if side == "left" else -1
            for lane_idx, lane in enumerate(lanes):
                try:
                    original_id = int(lane.get("id"))
                except (TypeError, ValueError):
                    continue

                lane["__orig_id"] = original_id
                new_id = (lane_idx + 1) * sign
                lane["id"] = new_id
                occurrence[(sec_idx, original_id)] = new_id

            if side == "left":
                lanes.sort(key=lambda item: float(item.get("id", 0)))
            else:
                lanes.sort(key=lambda item: float(item.get("id", 0)), reverse=True)

    def _lookup(section_index: int, target: int, direction: int) -> int:
        idx = section_index + direction
        while 0 <= idx < len(lane_sections):
            mapped = occurrence.get((idx, target))
            if mapped is not None:
                return mapped
            idx += direction
        mapped = occurrence.get((section_index, target))
        if mapped is not None:
            return mapped
        return target

    # Second pass: remap predecessor/successor identifiers using the new IDs.
    for sec_idx, section in enumerate(lane_sections):
        for side in ("left", "right"):
            lanes = section.get(side, []) or []
            for lane in lanes:
                original_id = lane.pop("__orig_id", None)
                preds = lane.get("predecessors") or []
                if preds:
                    remapped = []
                    for pid in preds:
                        try:
                            target = int(pid)
                        except (TypeError, ValueError):
                            remapped.append(pid)
                            continue
                        remapped.append(_lookup(sec_idx, target, direction=-1))
                    lane["predecessors"] = remapped

                succs = lane.get("successors") or []
                if succs:
                    remapped = []
                    for sid in succs:
                        try:
                            target = int(sid)
                        except (TypeError, ValueError):
                            remapped.append(sid)
                            continue
                        remapped.append(_lookup(sec_idx, target, direction=1))
                    lane["successors"] = remapped
