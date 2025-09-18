"""Helpers for building per-section lane specifications."""

from typing import Any, Dict, Iterable, List, Optional


def build_lane_spec(
    sections: Iterable[Dict[str, Any]],
    lane_topo: Optional[Dict[str, Any]],
    defaults: Dict[str, Any],
    lane_div_df,
) -> List[Dict[str, Any]]:
    """Return metadata for each lane section used by the writer."""

    lanes_guess = (lane_topo or {}).get("lanes_guess") or [1, -1]


    roadmark_type = "solid"
    if lane_div_df is not None and len(lane_div_df) > 0:
        from csv2xodr.mapping.core import mark_type_from_division_row

        roadmark_type = mark_type_from_division_row(lane_div_df.iloc[0])

    sections_list = list(sections)
    total = len(sections_list)

    out: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections_list):
        has_prev = idx > 0
        has_next = idx < total - 1
        out.append(
            {
                "s0": sec["s0"],
                "s1": sec["s1"],
                "lanes": lanes_guess,
                "lane_width": defaults.get("lane_width_m", 3.5),
                "roadMark": roadmark_type,
                "predecessor": has_prev,
                "successor": has_next,
            }
        )
    return out
