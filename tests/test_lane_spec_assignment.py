from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csv2xodr.lane_spec import build_lane_spec
from csv2xodr.simpletable import DataFrame
from csv2xodr.topology.core import build_lane_topology


def test_positive_lanes_remain_on_left_without_right_hints():
    rows = []
    for lane_no in (1, 2, 3):
        rows.append(
            {
                "Offset[cm]": "0",
                "End Offset[cm]": "100",
                "レーンID": f"G{lane_no}",
                "レーン番号": str(lane_no),
                "Lane Width[m]": "3.5",
            }
        )

    topo = build_lane_topology(DataFrame(rows))
    sections = [{"s0": 0.0, "s1": 10.0}]

    specs = build_lane_spec(sections, topo, defaults={}, lane_div_df=DataFrame([]))

    left_ids = [lane["id"] for lane in specs[0]["left"]]
    right_ids = [lane["id"] for lane in specs[0]["right"]]

    assert len(left_ids) == 3
    assert not right_ids
    assert all(lane_id > 0 for lane_id in left_ids)


def test_positive_lanes_ignore_lane_count_based_split():
    rows = []
    for lane_no in (1, 2, 3):
        rows.append(
            {
                "Offset[cm]": "0",
                "End Offset[cm]": "100",
                "レーンID": f"G{lane_no}",
                "レーン番号": str(lane_no),
                "Lane Width[m]": "3.5",
                "Lane Count": "6",
            }
        )

    topo = build_lane_topology(DataFrame(rows))
    sections = [{"s0": 0.0, "s1": 10.0}]

    specs = build_lane_spec(
        sections,
        topo,
        defaults={"default_lane_side": "right"},
        lane_div_df=DataFrame([]),
    )

    left_ids = [lane["id"] for lane in specs[0]["left"]]
    right_ids = [lane["id"] for lane in specs[0]["right"]]

    assert len(left_ids) == 3
    assert not right_ids
    assert all(lane_id > 0 for lane_id in left_ids)


def test_positive_lanes_with_sparse_numbers_stay_on_left():
    rows = []
    for lane_no in (1, 3, 5, 7):
        rows.append(
            {
                "Offset[cm]": "0",
                "End Offset[cm]": "100",
                "レーンID": f"G{lane_no}",
                "レーン番号": str(lane_no),
                "Lane Width[m]": "3.5",
                "Lane Count": "8",
            }
        )

    topo = build_lane_topology(DataFrame(rows))
    sections = [{"s0": 0.0, "s1": 10.0}]

    specs = build_lane_spec(
        sections,
        topo,
        defaults={"default_lane_side": "right"},
        lane_div_df=DataFrame([]),
    )

    left_ids = [lane["id"] for lane in specs[0]["left"]]
    right_ids = [lane["id"] for lane in specs[0]["right"]]

    assert len(left_ids) == 4
    assert not right_ids
    assert all(lane_id > 0 for lane_id in left_ids)


def test_lane_count_split_requires_right_evidence():
    rows = []
    for lane_no in (-2, -1, 1, 2):
        rows.append(
            {
                "Offset[cm]": "0",
                "End Offset[cm]": "100",
                "レーンID": f"G{lane_no}",
                "レーン番号": str(lane_no),
                "Lane Width[m]": "3.5",
                "Lane Count": "4",
            }
        )

    topo = build_lane_topology(DataFrame(rows))
    sections = [{"s0": 0.0, "s1": 10.0}]

    specs = build_lane_spec(sections, topo, defaults={}, lane_div_df=DataFrame([]))

    left_ids = [lane["id"] for lane in specs[0]["left"]]
    right_ids = [lane["id"] for lane in specs[0]["right"]]

    assert set(left_ids) == {1, 2}
    assert set(right_ids) == {-1, -2}
