from pathlib import Path

import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csv2xodr.lane_spec import build_lane_spec
from csv2xodr.simpletable import DataFrame


def _make_lane_topology(line_id: str):
    return {
        "lane_count": 1,
        "groups": {"A": ["A:1"]},
        "lanes": {
            "A:1": {
                "lane_no": 1,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 10.0,
                        "width": 3.5,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {2: line_id},
                    }
                ],
            }
        },
    }


def test_lane_spec_attaches_geometry_and_clips():
    sections = [{"s0": 0.0, "s1": 4.0}, {"s0": 4.0, "s1": 10.0}]

    lane_div = DataFrame(
        [
            {
                "区画線ID": "100",
                "Offset[cm]": "0",
                "End Offset[cm]": "1000",
                "始点側線幅[cm]": "12",
                "終点側線幅[cm]": "12",
                "種別": "1",
                "Is Retransmission": "false",
            }
        ]
    )

    line_geometry_lookup = {
        "100": [
            {
                "s": [0.0, 5.0, 10.0],
                "x": [0.0, 5.0, 10.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
            }
        ]
    }

    specs = build_lane_spec(
        sections,
        _make_lane_topology("100"),
        defaults={},
        lane_div_df=lane_div,
        line_geometry_lookup=line_geometry_lookup,
    )

    first_lane = specs[0]["left"][0]
    assert first_lane["roadMark"]["type"] == "solid"
    assert first_lane["roadMark"]["width"] == pytest.approx(0.12)
    geom_first = first_lane["roadMark"].get("geometry")
    assert geom_first is not None
    assert geom_first["s"][0] == pytest.approx(0.0)
    assert geom_first["s"][-1] == pytest.approx(4.0)
    assert geom_first["x"][-1] == pytest.approx(4.0)

    second_lane = specs[1]["left"][0]
    geom_second = second_lane["roadMark"]["geometry"]
    assert geom_second["s"][0] == pytest.approx(4.0)
    assert geom_second["s"][-1] == pytest.approx(10.0)
    assert geom_second["x"][0] == pytest.approx(4.0)
