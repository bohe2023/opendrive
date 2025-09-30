"""Helpers for building per-section lane specifications."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from csv2xodr.mapping.core import mark_type_from_division_row

from csv2xodr.simpletable import DataFrame
from csv2xodr.topology.core import _canonical_numeric


def _clip_geometry_segment(geom: Dict[str, List[float]], s0: float, s1: float) -> Optional[Dict[str, List[float]]]:
    s_vals = geom.get("s") or []
    if not s_vals:
        return None

    x_vals = geom.get("x") or []
    y_vals = geom.get("y") or []
    z_vals = geom.get("z") or []

    if len(s_vals) != len(x_vals) or len(s_vals) != len(y_vals) or len(s_vals) != len(z_vals):
        return None

    def _interpolate(a: float, b: float, ta: float, tb: float, target: float) -> float:
        if tb == ta:
            return a
        t = (target - ta) / (tb - ta)
        return a + t * (b - a)

    clipped: List[Tuple[float, float, float, float]] = []

    def _append(point: Tuple[float, float, float, float]) -> None:
        if not clipped or abs(clipped[-1][0] - point[0]) > 1e-6:
            clipped.append(point)

    for idx in range(len(s_vals) - 1):
        sa = s_vals[idx]
        sb = s_vals[idx + 1]
        xa, xb = x_vals[idx], x_vals[idx + 1]
        ya, yb = y_vals[idx], y_vals[idx + 1]
        za, zb = z_vals[idx], z_vals[idx + 1]

        if sb <= s0 or sa >= s1:
            continue

        if sa < s0 <= sb:
            x_new = _interpolate(xa, xb, sa, sb, s0)
            y_new = _interpolate(ya, yb, sa, sb, s0)
            z_new = _interpolate(za, zb, sa, sb, s0)
            _append((s0, x_new, y_new, z_new))

        if s0 <= sa <= s1:
            _append((sa, xa, ya, za))

        if sa < s1 <= sb:
            x_new = _interpolate(xa, xb, sa, sb, s1)
            y_new = _interpolate(ya, yb, sa, sb, s1)
            z_new = _interpolate(za, zb, sa, sb, s1)
            _append((s1, x_new, y_new, z_new))
        elif s0 <= sb <= s1:
            _append((sb, xb, yb, zb))

    if not clipped:
        if len(s_vals) == 1 and s0 <= s_vals[0] <= s1:
            clipped.append((s_vals[0], x_vals[0], y_vals[0], z_vals[0]))

    if not clipped:
        return None

    return {
        "s": [p[0] for p in clipped],
        "x": [p[1] for p in clipped],
        "y": [p[2] for p in clipped],
        "z": [p[3] for p in clipped],
    }


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


def _build_division_lookup(lane_div_df: Optional[DataFrame],
                           line_geometry_lookup: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                           offset_mapper=None) -> Dict[str, Dict[str, Any]]:
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
) -> List[Dict[str, Any]]:
    """Return metadata for each lane section used by the writer."""

    sections_list = list(sections)
    lane_info = (lane_topo or {}).get("lanes") or {}
    lane_groups = (lane_topo or {}).get("groups") or {}
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

    division_lookup = _build_division_lookup(
        lane_div_df, line_geometry_lookup=line_geometry_lookup, offset_mapper=offset_mapper
    )

    base_ids = sorted(lane_groups.keys())
    lanes_per_base = {base: len(lane_groups[base]) for base in base_ids}

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

    if positive_bases or negative_bases:
        # When the input data contains explicit lane number signs we rely on
        # them to determine the side of the reference line.  This avoids
        # mis-classifying sequential lane groups that belong to the same
        # carriageway (a frequent pattern in MPUs where lane IDs change along
        # the path but the driving direction does not).
        left_bases = sorted(set(positive_bases), key=lambda b: base_ids.index(b))
        right_bases = sorted(set(negative_bases), key=lambda b: base_ids.index(b))
    else:
        target_left = lane_count // 2 if lane_count else sum(lanes_per_base.values()) // 2
        left_bases = []
        acc = 0
        for base in base_ids:
            if acc < target_left:
                left_bases.append(base)
                acc += lanes_per_base[base]
        right_bases = [base for base in base_ids if base not in left_bases]
    if not left_bases and base_ids:
        left_bases = base_ids[:1]
        right_bases = [base for base in base_ids if base not in left_bases]

    lane_id_map: Dict[str, int] = {}
    lane_side_map: Dict[str, str] = {}

    left_id_by_lane_no: Dict[Tuple[Optional[str], int], int] = {}
    current_id = 1
    for base in left_bases:
        ordered = reversed(sorted(lane_groups.get(base, []), key=lambda x: lane_info[x]["lane_no"]))
        for uid in ordered:
            info = lane_info[uid]
            lane_no = info["lane_no"]
            key = (info.get("base_id"), lane_no)
            assigned = left_id_by_lane_no.get(key)
            if assigned is None:
                assigned = current_id
                left_id_by_lane_no[key] = assigned
                current_id += 1
            lane_id_map[uid] = assigned
            lane_side_map[uid] = "left"

    right_id_by_lane_no: Dict[Tuple[Optional[str], int], int] = {}
    current_id = -1
    for base in right_bases:
        ordered = reversed(sorted(lane_groups.get(base, []), key=lambda x: lane_info[x]["lane_no"]))
        for uid in ordered:
            info = lane_info[uid]
            lane_no = info["lane_no"]
            key = (info.get("base_id"), lane_no)
            assigned = right_id_by_lane_no.get(key)
            if assigned is None:
                assigned = current_id
                right_id_by_lane_no[key] = assigned
                current_id -= 1
            lane_id_map[uid] = assigned
            lane_side_map[uid] = "right"

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
            line_id = segment.get("line_positions", {}).get(pos_key)
            if not line_id:
                line_id = segment.get("line_positions", {}).get(alt_key)

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

        out.append({"s0": s0, "s1": s1, "left": section_left, "right": section_right})

    return out


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
