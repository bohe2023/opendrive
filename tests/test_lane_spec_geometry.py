from pathlib import Path

import math
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csv2xodr.lane_spec import (
    build_lane_spec,
    _build_division_lookup,
    _compute_lane_offset,
    _estimate_lane_side_from_geometry,
)
from csv2xodr.simpletable import DataFrame
from csv2xodr.writer.xodr_writer import write_xodr


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


def test_build_division_lookup_prefers_true_retransmission_segments():
    lane_div = DataFrame(
        [
            {
                "区画線ID": "100",
                "Offset[cm]": "0",
                "End Offset[cm]": "100",
                "種別": "1",
                "Is Retransmission": "false",
            },
            {
                "区画線ID": "100",
                "Offset[cm]": "0",
                "End Offset[cm]": "100",
                "種別": "3",
                "Is Retransmission": "true",
            },
        ]
    )

    lookup = _build_division_lookup(lane_div)

    segments = lookup["100"]["segments"]
    assert len(segments) == 1
    assert segments[0]["type"] == "broken", "segment should reflect the true retransmission row"
    assert "_is_retrans" not in segments[0]


def test_strong_geometry_hint_overrides_lane_numbers():
    sections = [{"s0": 0.0, "s1": 10.0}]

    lane_topology = {
        "lane_count": 0,
        "groups": {"G": ["G:1"]},
        "lanes": {
            "G:1": {
                "lane_no": 1,
                "segments": [
                    {
                        "uid": "G:1",
                        "base_id": "G",
                        "lane_no": 1,
                        "start": 0.0,
                        "end": 10.0,
                        "width": 3.5,
                        "left_neighbor": None,
                        "right_neighbor": None,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {1: "L1", 2: "L1"},
                        "is_retrans": False,
                        "lane_kind": "driving",
                    }
                ],
            }
        },
    }

    lane_div = DataFrame(
        [
            {
                "区画線ID": "L1",
                "Offset[cm]": "0",
                "End Offset[cm]": "1000",
                "始点側線幅[cm]": "12",
                "終点側線幅[cm]": "12",
                "区画線種別": "1",
                "Is Retransmission": "false",
            }
        ]
    )

    line_geometry_lookup = {
        "L1": [
            {
                "s": [0.0, 10.0],
                "x": [0.0, 10.0],
                "y": [-1.0, -1.0],
                "z": [0.0, 0.0],
            }
        ]
    }

    meters_to_degrees = 180.0 / (math.pi * 6378137.0)
    lanes_geometry = DataFrame(
        [
            {
                "Lane ID": "G",
                "Latitude": -3.0 * meters_to_degrees,
                "Longitude": 0.0,
                "Offset[cm]": 0.0,
            },
            {
                "Lane ID": "G",
                "Latitude": -3.0 * meters_to_degrees,
                "Longitude": 10.0 * meters_to_degrees,
                "Offset[cm]": 1000.0,
            },
        ]
    )

    centerline = DataFrame(
        {
            "s": [0.0, 10.0],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "hdg": [0.0, 0.0],
        }
    )

    specs = build_lane_spec(
        sections,
        lane_topology,
        defaults={"lane_width_m": 3.5},
        lane_div_df=lane_div,
        line_geometry_lookup=line_geometry_lookup,
        offset_mapper=lambda value: float(value),
        lanes_geometry_df=lanes_geometry,
        centerline=centerline,
        geo_origin=(0.0, 0.0),
    )

    assert not specs[0]["left"], "strong right-side geometry should prevent left assignment"
    assert len(specs[0]["right"]) == 1


def test_geometry_hint_recenters_common_bias():
    meters_to_degrees = 180.0 / (math.pi * 6378137.0)

    centerline = DataFrame(
        {
            "s": [0.0, 10.0],
            "x": [0.0, 0.0],
            "y": [0.0, 10.0],
            "hdg": [math.pi / 2, math.pi / 2],
        }
    )

    rows = []
    for lane_id, lateral in (("L", 1.0), ("M", 4.0), ("R", 7.0)):
        for offset_m in (0.0, 10.0):
            rows.append(
                {
                    "Lane ID": lane_id,
                    "Offset[cm]": str(offset_m * 100.0),
                    "緯度[deg]": offset_m * meters_to_degrees,
                    "経度[deg]": lateral * meters_to_degrees,
                }
            )

    lanes_geometry = DataFrame(rows)

    side_map, strength_map, bias_map = _estimate_lane_side_from_geometry(
        lanes_geometry,
        centerline,
        offset_mapper=lambda value: float(value),
        geo_origin=(0.0, 0.0),
        expected_side_counts=(1, 1),
    )

    assert side_map["L"] == "left"
    assert side_map["R"] == "right"
    assert "M" not in side_map or abs(strength_map.get("M", 0.0)) <= 1e-9
    assert bias_map["M"] == pytest.approx(0.0, abs=1e-6)


def test_positive_lanes_stay_on_single_side():
    sections = [{"s0": 0.0, "s1": 10.0}]

    lane_topology = {
        "lane_count": 3,
        "groups": {
            "A": ["A:1"],
            "B": ["B:1"],
            "C": ["C:1"],
        },
        "lanes": {
            "A:1": {
                "base_id": "A",
                "lane_no": 1,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 10.0,
                        "width": 3.5,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {},
                    }
                ],
            },
            "B:1": {
                "base_id": "B",
                "lane_no": 2,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 10.0,
                        "width": 3.5,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {},
                    }
                ],
            },
            "C:1": {
                "base_id": "C",
                "lane_no": 3,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 10.0,
                        "width": 3.5,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {},
                    }
                ],
            },
        },
    }

    specs = build_lane_spec(sections, lane_topology, defaults={}, lane_div_df=None)

    assert specs[0]["right"] == [], "positive-only lanes should not be inferred as right-side lanes"
    assert [lane["uid"] for lane in specs[0]["left"]] == ["A:1", "B:1", "C:1"]


def test_lane_offset_prefers_innermost_edges():
    lane_info = {
        "L:1": {"base_id": "L"},
        "R:1": {"base_id": "R1"},
        "R:2": {"base_id": "R2"},
    }

    section_left = [
        {
            "uid": "L:1",
            "width": 3.6,
        }
    ]
    section_right = [
        {
            "uid": "R:1",
            "width": 3.6,
        },
        {
            "uid": "R:2",
            "width": 3.6,
        },
    ]

    geometry_bias = {"L": 5.0, "R1": -3.0, "R2": -7.0}

    offset = _compute_lane_offset(section_left, section_right, lane_info, geometry_bias)

    assert offset == pytest.approx(-1.0)


def test_lane_offset_respects_central_lane_bias():
    lane_info = {
        "C:0": {"base_id": "C", "lane_no": 0},
        "L:1": {"base_id": "L", "lane_no": 1},
        "R:-1": {"base_id": "R", "lane_no": -1},
    }

    section_left = [
        {"uid": "L:1", "width": 3.5, "lane_no": 1},
        {"uid": "C:0", "width": 3.5, "lane_no": 0},
    ]
    section_right = [
        {"uid": "R:-1", "width": 3.5, "lane_no": -1},
    ]

    geometry_bias = {"L": 3.5, "C": 0.8, "R": -3.5}

    offset = _compute_lane_offset(section_left, section_right, lane_info, geometry_bias)

    assert offset == pytest.approx(-0.8)


def test_lane_offset_uses_bias_closest_to_reference_when_numbers_positive():
    lane_info = {
        "L:1": {"base_id": "L", "lane_no": 1},
        "M:2": {"base_id": "M", "lane_no": 2},
        "R:3": {"base_id": "R", "lane_no": 3},
    }

    section_left = [
        {"uid": "L:1", "width": 3.5, "lane_no": 1},
        {"uid": "M:2", "width": 3.5, "lane_no": 2},
    ]
    section_right = [
        {"uid": "R:3", "width": 3.5, "lane_no": 3},
    ]

    geometry_bias = {"L": 3.5, "M": 0.6, "R": -3.5}

    offset = _compute_lane_offset(section_left, section_right, lane_info, geometry_bias)

    assert offset == pytest.approx(-0.6)


def test_lane_offset_recenters_single_right_side_stack():
    lane_info = {
        "R:1": {"base_id": "R1"},
        "R:2": {"base_id": "R2"},
        "R:3": {"base_id": "R3"},
    }

    section_right = [
        {"uid": "R:1", "width": 3.5},
        {"uid": "R:2", "width": 3.5},
        {"uid": "R:3", "width": 3.5},
    ]

    geometry_bias = {"R1": -3.5, "R2": -7.0, "R3": -10.5}

    offset = _compute_lane_offset([], section_right, lane_info, geometry_bias)

    assert offset == pytest.approx(7.0)


def test_lane_offset_recenters_single_left_side_stack():
    lane_info = {
        "L:1": {"base_id": "L1"},
        "L:2": {"base_id": "L2"},
        "L:3": {"base_id": "L3"},
    }

    section_left = [
        {"uid": "L:1", "width": 3.5},
        {"uid": "L:2", "width": 3.5},
        {"uid": "L:3", "width": 3.5},
    ]

    geometry_bias = {"L1": 3.5, "L2": 7.0, "L3": 10.5}

    offset = _compute_lane_offset(section_left, [], lane_info, geometry_bias)

    assert offset == pytest.approx(-7.0)


def test_write_xodr_emits_explicit_lane_mark_geometry(tmp_path):
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

    lane_specs = build_lane_spec(
        sections,
        _make_lane_topology("100"),
        defaults={},
        lane_div_df=lane_div,
        line_geometry_lookup=line_geometry_lookup,
    )

    centerline = DataFrame(
        {
            "s": [0.0, 4.0, 10.0],
            "x": [0.0, 4.0, 10.0],
            "y": [0.0, 0.0, 0.0],
            "hdg": [0.0, 0.0, 0.0],
        }
    )

    out_file = Path(tmp_path) / "explicit_roadmark.xodr"
    write_xodr(centerline, sections, lane_specs, out_file)

    root = ET.parse(out_file).getroot()
    header = root.find("header")
    assert header is not None
    assert header.get("revMinor") == "4"
    assert header.get("version") == "1.4"

    road_marks = root.findall(".//roadMark")

    assert len(road_marks) == 2

    first = road_marks[0]
    explicit_first = first.find("explicit")
    assert explicit_first is not None
    first_geoms = explicit_first.findall("geometry")
    assert first_geoms

    attrs_first = first_geoms[0].attrib
    assert float(attrs_first["sOffset"]) == pytest.approx(0.0)
    assert float(attrs_first["x"]) == pytest.approx(0.0)
    assert float(attrs_first["y"]) == pytest.approx(0.0)
    assert float(attrs_first["z"]) == pytest.approx(0.0)
    assert float(attrs_first["length"]) == pytest.approx(4.0)

    last = road_marks[1]
    explicit_last = last.find("explicit")
    assert explicit_last is not None
    last_geoms = explicit_last.findall("geometry")
    assert len(last_geoms) == 2
    assert float(last_geoms[0].attrib["sOffset"]) == pytest.approx(0.0)
    assert float(last_geoms[0].attrib["x"]) == pytest.approx(4.0)
    assert float(last_geoms[-1].attrib["x"]) == pytest.approx(5.0)


def test_write_xodr_promotes_central_lane_to_reference(tmp_path):
    centerline = DataFrame(
        {
            "s": [0.0, 10.0],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "hdg": [0.0, 0.0],
        }
    )

    sections = [{"s0": 0.0, "s1": 10.0}]

    shared_mark = {"type": "solid", "width": 0.12, "laneChange": "none"}

    lane_specs = [
        {
            "s0": 0.0,
            "s1": 10.0,
            "left": [
                {
                    "id": 1,
                    "lane_no": 1,
                    "width": 3.5,
                    "roadMark": shared_mark,
                    "predecessors": [],
                    "successors": [],
                    "type": "driving",
                },
                {
                    "id": 2,
                    "lane_no": 0,
                    "width": 3.5,
                    "roadMark": shared_mark,
                    "predecessors": [],
                    "successors": [],
                    "type": "driving",
                },
            ],
            "right": [
                {
                    "id": -1,
                    "lane_no": -1,
                    "width": 3.5,
                    "roadMark": shared_mark,
                    "predecessors": [],
                    "successors": [],
                    "type": "driving",
                }
            ],
        }
    ]

    out_file = Path(tmp_path) / "center_lane_promoted.xodr"
    write_xodr(centerline, sections, lane_specs, out_file)

    root = ET.parse(out_file).getroot()
    center_lane = root.find(".//laneSection/center/lane")
    assert center_lane is not None
    assert center_lane.get("type") == "driving"

    widths = center_lane.findall("width")
    assert len(widths) == 1
    assert float(widths[0].attrib["a"]) == pytest.approx(3.5)

    road_mark = center_lane.find("roadMark")
    assert road_mark is not None
    assert road_mark.get("type") == "solid"

    left_lanes = root.findall(".//laneSection/left/lane")
    assert len(left_lanes) == 1
    assert left_lanes[0].get("id") == "1"
