"""Helpers that adapt US-specific CSV tables to the generic converter pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from csv2xodr.simpletable import DataFrame
from csv2xodr.topology.core import _canonical_numeric


def _find_column(columns: Iterable[str], *keywords: str) -> Optional[str]:
    lowered = [kw.lower() for kw in keywords]
    for col in columns:
        value = col.strip().lower()
        if all(keyword in value for keyword in lowered):
            return col
    return None


def _to_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return None
        return float(text)
    except Exception:
        return None


def merge_lane_width_into_links(
    lane_link_df: Optional[DataFrame], lane_width_df: Optional[DataFrame]
) -> Optional[DataFrame]:
    """Inject average lane widths (in centimetres) into the link table."""

    if lane_link_df is None or len(lane_link_df) == 0:
        return lane_link_df
    if lane_width_df is None or len(lane_width_df) == 0:
        return lane_link_df

    width_lane_col = _find_column(lane_width_df.columns, "lane", "number")
    width_value_col = None
    for col in lane_width_df.columns:
        lowered = col.strip().lower()
        if "width" in lowered or "幅員" in col:
            width_value_col = col
            break

    if width_lane_col is None or width_value_col is None:
        return lane_link_df

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for idx in range(len(lane_width_df)):
        row = lane_width_df.iloc[idx]
        lane_raw = row[width_lane_col]
        lane_id = _canonical_numeric(lane_raw, allow_negative=True)
        if lane_id is None:
            lane_id = str(lane_raw).strip()
        if not lane_id:
            continue

        width_val = _to_float(row[width_value_col])
        if width_val is None or width_val <= 0:
            continue

        totals[lane_id] = totals.get(lane_id, 0.0) + width_val
        counts[lane_id] = counts.get(lane_id, 0) + 1

    if not totals:
        return lane_link_df

    width_map = {key: totals[key] / counts[key] for key in totals}

    link_lane_col = _find_column(lane_link_df.columns, "lane", "number")
    if link_lane_col is None:
        link_lane_col = _find_column(lane_link_df.columns, "レーン", "番号")
    if link_lane_col is None:
        return lane_link_df

    existing_columns = list(lane_link_df.columns)
    width_col_name = "幅員"
    if width_col_name not in existing_columns:
        columns = existing_columns + [width_col_name]
    else:
        columns = existing_columns

    new_rows = []
    for idx in range(len(lane_link_df)):
        src = lane_link_df.iloc[idx]
        row = {col: src[col] for col in existing_columns}

        lane_raw = src[link_lane_col]
        lane_id = _canonical_numeric(lane_raw, allow_negative=True)
        if lane_id is None:
            lane_id = str(lane_raw).strip()

        width_m = width_map.get(lane_id) if lane_id else None
        if width_m is None and lane_raw in width_map:
            width_m = width_map[lane_raw]

        if width_m is not None:
            row[width_col_name] = width_m * 100.0

        new_rows.append(row)

    return DataFrame(new_rows, columns=columns)
