"""Helpers for building per-section lane specifications."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from csv2xodr.mapping.core import mark_type_from_division_row

from csv2xodr.simpletable import DataFrame


def _lookup_line_segment(segments: List[Dict[str, Any]], s0: float, s1: float) -> Optional[Dict[str, Any]]:
    for seg in segments:
        if seg["s1"] <= s0:
            continue
        if seg["s0"] >= s1:
            continue
        return seg
    return segments[0] if segments else None


def _build_division_lookup(lane_div_df: Optional[DataFrame],
                           offset_mapper=None) -> Dict[str, List[Dict[str, Any]]]:
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
    start_w_col = find_col("始点側線幅")
    end_w_col = find_col("終点側線幅")
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
        line_id = str(line_id_raw).strip()
        if line_id in {"", "0", "0.0", "-1", "-1.0"}:
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

    records: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    for record in raw_records:
        adj_start = record["start"] - base_offset_m
        adj_end = record["end"] - base_offset_m
        key = (record["line_id"], adj_start, adj_end)

        mapped_start = adj_start
        mapped_end = adj_end
        if offset_mapper is not None:
            mapped_start = offset_mapper(mapped_start)
            mapped_end = offset_mapper(mapped_end)

        data = {
            "row": record["row"],
            "width": record["width"],
            "is_retrans": record["is_retrans"],
            "s0": mapped_start,
            "s1": mapped_end,
        }

        existing = records.get(key)
        if existing is not None:
            if existing["is_retrans"] and not record["is_retrans"]:
                records[key] = data
            continue

        records[key] = data

    grouped: Dict[str, List[Tuple[float, float, Dict[str, Any]]]] = {}
    for (line_id, start, end), data in records.items():
        grouped.setdefault(line_id, []).append((start, end, data))

    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for line_id, segments in grouped.items():
        segments.sort(key=lambda item: (item[0], item[1]))
        cleaned: List[Dict[str, Any]] = []
        for start, end, data in segments:
            if cleaned:
                prev = cleaned[-1]
                if abs(prev["s0"] - start) < 1e-6 and abs(prev["s1"] - end) < 1e-6:
                    continue
                if start < prev["s1"]:
                    start = max(prev["s1"], start)
                    if start >= end:
                        continue

            mark_type = mark_type_from_division_row(data["row"])
            cleaned.append({
                "s0": data.get("s0", start),
                "s1": data.get("s1", end),
                "type": mark_type,
                "width": data["width"],
            })

        if cleaned:
            lookup[line_id] = cleaned

    return lookup



def build_lane_spec(
    sections: Iterable[Dict[str, Any]],
    lane_topo: Optional[Dict[str, Any]],
    defaults: Dict[str, Any],
    lane_div_df,
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

    division_lookup = _build_division_lookup(lane_div_df, offset_mapper)

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

    left_id_by_lane_no: Dict[int, int] = {}
    current_id = 1
    for base in left_bases:
        ordered = reversed(sorted(lane_groups.get(base, []), key=lambda x: lane_info[x]["lane_no"]))
        for uid in ordered:
            lane_no = lane_info[uid]["lane_no"]
            assigned = left_id_by_lane_no.get(lane_no)
            if assigned is None:
                assigned = current_id
                left_id_by_lane_no[lane_no] = assigned
                current_id += 1
            lane_id_map[uid] = assigned
            lane_side_map[uid] = "left"

    right_id_by_lane_no: Dict[int, int] = {}
    current_id = -1
    for base in right_bases:
        ordered = reversed(sorted(lane_groups.get(base, []), key=lambda x: lane_info[x]["lane_no"]))
        for uid in ordered:
            lane_no = lane_info[uid]["lane_no"]
            assigned = right_id_by_lane_no.get(lane_no)
            if assigned is None:
                assigned = current_id
                right_id_by_lane_no[lane_no] = assigned
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

            predecessors: List[int] = []
            if order > 0:
                predecessors.append(lane_id)
            else:
                for target in segment.get("predecessors", []):
                    mapped = lane_id_map.get(target)
                    if mapped is not None:
                        predecessors.append(mapped)

            successors: List[int] = []
            if order < len(lane_indices) - 1:
                successors.append(lane_id)
            else:
                for target in segment.get("successors", []):
                    mapped = lane_id_map.get(target)
                    if mapped is not None:
                        successors.append(mapped)

            pos_key = 2 if side == "left" else 1
            alt_key = 1 if side == "left" else 2
            line_id = segment.get("line_positions", {}).get(pos_key)
            if not line_id:
                line_id = segment.get("line_positions", {}).get(alt_key)

            mark = None
            if line_id:
                mark_segment = _lookup_line_segment(division_lookup.get(line_id, []), s0, s1)
                if mark_segment:
                    mark_width = mark_segment.get("width") or 0.12
                    lane_change = "both" if mark_segment.get("type") != "solid" else "none"
                    mark = {
                        "type": mark_segment.get("type") or "solid",
                        "width": mark_width,
                        "laneChange": lane_change,
                    }

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
            }

            if side == "left":
                section_left.append(lane_entry)
            else:
                section_right.append(lane_entry)

        section_left.sort(key=lambda item: item["id"])
        section_right.sort(key=lambda item: item["id"], reverse=True)

        out.append({"s0": s0, "s1": s1, "left": section_left, "right": section_right})

    return out
