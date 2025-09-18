from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from csv2xodr.simpletable import DataFrame, Series


def _canonical_numeric(value: Any, *, allow_negative: bool = False) -> Optional[str]:
    """Return a stable string representation for identifier-like values."""

    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s in {"0", "0.0"}:
        return None
    if not allow_negative and s in {"-1", "-1.0"}:
        return None
    try:
        dec = Decimal(s)
    except InvalidOperation:
        return s
    if dec == 0:
        return None
    if not allow_negative and dec == -1:
        return None
    if dec == int(dec):
        return str(int(dec))
    return format(dec.normalize(), "f")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(Decimal(s))
    except (InvalidOperation, ValueError):
        return None


def _cm_to_m(value: Any) -> Optional[float]:
    val = _to_float(value)
    if val is None:
        return None
    return val / 100.0


def _compose_lane_uid(base_id: str, lane_no: int) -> str:
    return f"{base_id}:{lane_no}"


def _find_column(columns: List[str], *keywords: str) -> Optional[str]:
    for col in columns:
        stripped = col.strip()
        if all(keyword in stripped for keyword in keywords):
            return col
    return None

def _offset_series(df: DataFrame):
    if df is None or len(df) == 0:
        return None, None
    off_cols = [c for c in df.columns if "Offset" in c]             # Offset[cm]
    end_cols = [c for c in df.columns if "End Offset" in c]         # End Offset[cm]
    off = None
    end = None
    base_offset = None
    if off_cols:
        off_values = df[off_cols[0]].astype(float).to_list()
        off_m = [v / 100.0 for v in off_values]
        base_offset = min(off_m) if off_m else 0.0
        off = Series([v - base_offset for v in off_m], name=off_cols[0], kind="column")
    if end_cols:
        end_values = df[end_cols[0]].astype(float).to_list()
        end_m = [v / 100.0 for v in end_values]
        if base_offset is None:
            base_offset = min(end_m) if end_m else 0.0
        end = Series([v - base_offset for v in end_m], name=end_cols[0], kind="column")
    return off, end

def make_sections(centerline: DataFrame,
                  lane_link_df: DataFrame = None,
                  lane_div_df: DataFrame = None,
                  min_len: float = 0.01):
    """
    Collect split points from lane_link/lane_div offsets (in meters),
    produce lane sections [s0, s1).
    """
    splits = set([0.0, float(centerline["s"].iloc[-1])])

    for df in (lane_link_df, lane_div_df):
        off, end = _offset_series(df)
        if off is not None:
            for v in off.values:
                splits.add(float(v))
        if end is not None:
            for v in end.values:
                splits.add(float(v))

    splits = sorted(splits)
    sections = []
    for i in range(len(splits) - 1):
        s0, s1 = splits[i], splits[i + 1]
        if s1 - s0 > min_len:
            sections.append({"s0": s0, "s1": s1})
    return sections

def build_lane_topology(lane_link_df: DataFrame) -> Dict[str, Any]:
    """Parse lane link CSV into lane-centric topology information."""

    if lane_link_df is None or len(lane_link_df) == 0:
        return {"lanes": {}, "groups": {}, "lane_count": 0}

    cols = list(lane_link_df.columns)

    start_col = _find_column(cols, "Offset")
    end_col = _find_column(cols, "End", "Offset")
    lane_id_col = _find_column(cols, "レーンID") or _find_column(cols, "Lane", "ID")
    lane_no_col = _find_column(cols, "レーン番号") or _find_column(cols, "Lane", "番号")
    lane_count_col = _find_column(cols, "Lane", "Number")
    width_col = _find_column(cols, "幅員")
    is_retrans_col = _find_column(cols, "Is", "Retransmission")
    left_col = _find_column(cols, "左側車線")
    right_col = _find_column(cols, "右側車線")
    forward_cols = [c for c in cols if "前方レーンID" in c and "数" not in c]
    backward_cols = [c for c in cols if "後方レーンID" in c and "数" not in c]

    line_id_cols: List[Tuple[int, str]] = []
    line_pos_cols: Dict[int, str] = {}
    for col in cols:
        stripped = col.strip()
        if "ライン型地物ID" in stripped:
            try:
                idx = int(stripped.split("(")[1].split(")")[0])
            except Exception:
                continue
            line_id_cols.append((idx, col))
        if "位置種別" in stripped:
            try:
                idx = int(stripped.split("(")[1].split(")")[0])
            except Exception:
                continue
            line_pos_cols[idx] = col

    line_id_cols.sort(key=lambda x: x[0])

    lane_count = 0
    if lane_count_col is not None:
        try:
            lane_count = int(float(lane_link_df[lane_count_col].iloc[0]))
        except Exception:
            lane_count = 0

    raw_records: List[Dict[str, Any]] = []
    for i in range(len(lane_link_df)):
        row = lane_link_df.iloc[i]

        start = _cm_to_m(row[start_col]) if start_col else None
        end = _cm_to_m(row[end_col]) if end_col else None
        if start is None or end is None:
            continue
        if end <= start:
            continue

        base_id = _canonical_numeric(row[lane_id_col]) if lane_id_col else None
        lane_no_val = row[lane_no_col] if lane_no_col else None
        lane_no = None
        if lane_no_val is not None:
            try:
                lane_no = int(float(str(lane_no_val).strip()))
            except Exception:
                lane_no = None
        if base_id is None or lane_no is None:
            continue

        width = _cm_to_m(row[width_col]) if width_col else None
        is_retrans = False
        if is_retrans_col:
            is_retrans = str(row[is_retrans_col]).strip().lower() == "true"

        left_neighbor = _canonical_numeric(row[left_col]) if left_col else None
        right_neighbor = _canonical_numeric(row[right_col]) if right_col else None

        forward_targets = []
        for col in forward_cols:
            tid = _canonical_numeric(row[col])
            if tid:
                forward_targets.append(tid)
        backward_targets = []
        for col in backward_cols:
            tid = _canonical_numeric(row[col])
            if tid:
                backward_targets.append(tid)

        line_positions: Dict[int, str] = {}
        for idx, col in line_id_cols:
            lid = _canonical_numeric(row[col], allow_negative=True)
            if not lid or lid in {"-1"}:
                continue
            pos_col = line_pos_cols.get(idx)
            pos_val = row[pos_col] if pos_col else None
            try:
                pos = int(float(str(pos_val).strip())) if pos_val is not None else None
            except Exception:
                pos = None
            if pos is None:
                continue
            if pos not in line_positions:
                line_positions[pos] = lid

        uid = _compose_lane_uid(base_id, lane_no)
        raw_records.append({
            "uid": uid,
            "base_id": base_id,
            "lane_no": lane_no,
            "start": start,
            "end": end,
            "width": width,
            "left_neighbor": left_neighbor,
            "right_neighbor": right_neighbor,
            "successors": forward_targets,
            "predecessors": backward_targets,
            "line_positions": line_positions,
            "is_retrans": is_retrans,
        })

    if not raw_records:
        return {"lanes": {}, "groups": {}, "lane_count": lane_count}

    base_offset_m = min(record["start"] for record in raw_records)

    records: Dict[Tuple[str, int, float, float], Dict[str, Any]] = {}
    for record in raw_records:
        adj_start = record["start"] - base_offset_m
        adj_end = record["end"] - base_offset_m

        key = (record["base_id"], record["lane_no"], adj_start, adj_end)
        updated = dict(record)
        updated["start"] = adj_start
        updated["end"] = adj_end

        existing = records.get(key)
        if existing is not None:
            if existing.get("is_retrans") and not record["is_retrans"]:
                records[key] = updated
            continue

        records[key] = updated

    lanes: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, List[str]] = {}

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for record in records.values():
        key = (record["base_id"], record["lane_no"])
        grouped.setdefault(key, []).append(record)

    for (base_id, lane_no), items in grouped.items():
        items.sort(key=lambda r: (r["start"], r["end"]))
        cleaned: List[Dict[str, Any]] = []
        for item in items:
            if cleaned:
                prev = cleaned[-1]
                if abs(item["start"] - prev["start"]) < 1e-6 and abs(item["end"] - prev["end"]) < 1e-6:
                    continue
                if item["start"] < prev["end"]:
                    adjusted_start = max(prev["end"], item["start"])
                    if adjusted_start >= item["end"]:
                        continue
                    item = dict(item)
                    item["start"] = adjusted_start
            cleaned.append(item)

        if not cleaned:
            continue

        uid = _compose_lane_uid(base_id, lane_no)
        lanes[uid] = {
            "base_id": base_id,
            "lane_no": lane_no,
            "segments": cleaned,
        }
        groups.setdefault(base_id, []).append(uid)

    for lane_list in groups.values():
        lane_list.sort(key=lambda uid: lanes[uid]["lane_no"])

    # resolve successor/predecessor ids into lane uids where possible
    all_uids = set(lanes.keys())
    for lane in lanes.values():
        for segment in lane["segments"]:
            lane_no = lane["lane_no"]
            resolved_successors = []
            for target in segment["successors"]:
                candidate = _compose_lane_uid(target, lane_no) if target else None
                if candidate and candidate in all_uids:
                    resolved_successors.append(candidate)
            segment["successors"] = resolved_successors

            resolved_predecessors = []
            for target in segment["predecessors"]:
                candidate = _compose_lane_uid(target, lane_no) if target else None
                if candidate and candidate in all_uids:
                    resolved_predecessors.append(candidate)
            segment["predecessors"] = resolved_predecessors

    return {"lanes": lanes, "groups": groups, "lane_count": lane_count}
