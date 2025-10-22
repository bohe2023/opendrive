import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csv2xodr.lane_spec import build_lane_spec
from csv2xodr.writer.xodr_writer import write_xodr


class _Series:
    def __init__(self, values):
        self._values = list(values)
        self.iloc = self

    def __getitem__(self, idx):
        return self._values[idx]


class _SimpleCenterline:
    def __init__(self, columns):
        self._columns = {k: list(v) for k, v in columns.items()}
        lengths = {len(v) for v in self._columns.values()}
        if len(lengths) != 1:
            raise ValueError("All columns must have the same length")

    def __len__(self):
        return len(next(iter(self._columns.values())))

    def __getitem__(self, item):
        return _Series(self._columns[item])


def _make_centerline():
    return _SimpleCenterline({
        "s": [0.0, 10.0],
        "x": [0.0, 10.0],
        "y": [0.0, 0.0],
        "hdg": [0.0, 0.0],
    })


def _simple_lane_topology(length: float):
    return {
        "lane_count": 2,
        "groups": {
            "L": ["L:1"],
            "R": ["R:1"],
        },
        "lanes": {
            "L:1": {
                "base_id": "L",
                "lane_no": 1,
                "segments": [
                    {
                        "start": 0.0,
                        "end": length,
                        "width": 3.5,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {},
                    }
                ],
            },
            "R:1": {
                "base_id": "R",
                "lane_no": -1,
                "segments": [
                    {
                        "start": 0.0,
                        "end": length,
                        "width": 3.5,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {},
                    }
                ],
            },
        },
    }


def test_lane_spec_flags_and_writer_links(tmp_path):
    sections = [
        {"s0": 0.0, "s1": 5.0},
        {"s0": 5.0, "s1": 10.0},
    ]

    lane_topology = {
        "lane_count": 2,
        "groups": {
            "A": ["A:1"],
            "B": ["B:1"],
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
                "lane_no": -1,
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

    lane_specs = build_lane_spec(sections, lane_topology, defaults={}, lane_div_df=None)

    assert lane_specs[0]["left"][0]["successors"] == [1]
    assert lane_specs[0]["left"][0]["predecessors"] == []
    assert lane_specs[1]["left"][0]["predecessors"] == [1]
    assert lane_specs[1]["left"][0]["successors"] == []

    out_file = Path(tmp_path) / "lane_links.xodr"
    write_xodr(_make_centerline(), sections, lane_specs, out_file)

    root = ET.parse(out_file).getroot()
    lane_sections = root.find(".//lanes").findall("laneSection")

    first_left_lane = lane_sections[0].find("./left/lane")
    assert first_left_lane is not None
    assert first_left_lane.attrib["id"] == "1"

    first_left_link = lane_sections[0].find("./left/lane/link")
    assert first_left_link is not None
    assert first_left_link.find("predecessor") is None
    assert first_left_link.find("successor").attrib["id"] == "1"

    right_lanes = lane_sections[-1].findall("./right/lane")
    assert right_lanes, "expected at least one right lane"
    assert right_lanes[0].attrib["id"] == "-1"

    last_right_link = right_lanes[0].find("link")
    assert last_right_link is not None
    assert last_right_link.find("predecessor").attrib["id"] == "-1"
    assert last_right_link.find("successor") is None


def test_lane_spec_keeps_positive_lanes_on_default_side_without_right_evidence():
    """All positive lane numbers without right-side hints remain on the default side."""
    sections = [{"s0": 0.0, "s1": 10.0}]

    lane_topology = {
        "lane_count": 4,
        "groups": {
            "A": ["A:1"],
            "B": ["B:2"],
            "C": ["C:3"],
            "D": ["D:4"],
        },
        "lanes": {
            "A:1": {
                "base_id": "A",
                "lane_no": 1,
                "segments": [{"start": 0.0, "end": 10.0, "width": 3.5, "successors": [], "predecessors": [], "line_positions": {}}],
            },
            "B:2": {
                "base_id": "B",
                "lane_no": 2,
                "segments": [{"start": 0.0, "end": 10.0, "width": 3.5, "successors": [], "predecessors": [], "line_positions": {}}],
            },
            "C:3": {
                "base_id": "C",
                "lane_no": 3,
                "segments": [{"start": 0.0, "end": 10.0, "width": 3.5, "successors": [], "predecessors": [], "line_positions": {}}],
            },
            "D:4": {
                "base_id": "D",
                "lane_no": 4,
                "segments": [{"start": 0.0, "end": 10.0, "width": 3.5, "successors": [], "predecessors": [], "line_positions": {}}],
            },
        },
    }

    defaults = {"default_lane_side": "right"}
    specs = build_lane_spec(
        sections, lane_topology, defaults=defaults, lane_div_df=None
    )

    assert len(specs) == 1
    section = specs[0]
    left_ids = {lane["id"] for lane in section["left"]}
    right_ids = {lane["id"] for lane in section["right"]}

    assert left_ids == {1, 2, 3, 4}
    assert right_ids == set()


def test_lane_spec_assigns_negative_lanes_to_right_when_present():
    sections = [{"s0": 0.0, "s1": 10.0}]

    lane_topology = {
        "lane_count": 4,
        "groups": {
            "A": ["A:1"],
            "B": ["B:2"],
            "C": ["C:3"],
            "D": ["D:-1"],
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
            "B:2": {
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
            "C:3": {
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
            "D:-1": {
                "base_id": "D",
                "lane_no": -1,
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

    assert len(specs) == 1
    section = specs[0]
    left_ids = {lane["id"] for lane in section["left"]}
    right_ids = {lane["id"] for lane in section["right"]}

    assert left_ids == {1, 2, 3}
    assert right_ids == {-1}


def test_lane_spec_balances_positive_and_negative_lane_numbers():
    sections = [{"s0": 0.0, "s1": 10.0}]

    lane_topology = {
        "lane_count": 4,
        "groups": {
            "A": ["A:1"],
            "B": ["B:2"],
            "C": ["C:-1"],
            "D": ["D:-2"],
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
            "B:2": {
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
            "C:-1": {
                "base_id": "C",
                "lane_no": -1,
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
            "D:-2": {
                "base_id": "D",
                "lane_no": -2,
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

    assert len(specs) == 1
    assert [lane["id"] for lane in specs[0]["left"]] == [1, 2]
    assert [lane["id"] for lane in specs[0]["right"]] == [-1, -2]


def test_lane_spec_uses_lane_count_when_only_positive_lane_numbers():
    """Lane count should not force right-side lanes without right evidence."""
    sections = [{"s0": 0.0, "s1": 10.0}]

    lane_topology = {
        "lane_count": 2,
        "groups": {
            "A": ["A:1"],
            "B": ["B:2"],
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
            "B:2": {
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
        },
    }

    specs = build_lane_spec(sections, lane_topology, defaults={}, lane_div_df=None)

    assert len(specs) == 1
    section = specs[0]
    left_ids = [lane["id"] for lane in section["left"]]

    assert left_ids == [1, 2]
    assert section["right"] == []


def test_lane_spec_splits_positive_lanes_when_neighbours_form_chain():
    """Left/right neighbours should reveal both sides even without negative lane numbers."""

    sections = [{"s0": 0.0, "s1": 10.0}]

    lane_topology = {
        "lane_count": 3,
        "groups": {
            "A": ["A:1"],
            "B": ["B:2"],
            "C": ["C:3"],
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
                        "left_neighbor": None,
                        "right_neighbor": "B",
                    }
                ],
            },
            "B:2": {
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
                        "left_neighbor": "A",
                        "right_neighbor": "C",
                    }
                ],
            },
            "C:3": {
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
                        "left_neighbor": "B",
                        "right_neighbor": None,
                    }
                ],
            },
        },
    }

    specs = build_lane_spec(sections, lane_topology, defaults={}, lane_div_df=None)

    assert len(specs) == 1
    section = specs[0]
    assert [lane["id"] for lane in section["left"]] == [1]
    assert [lane["id"] for lane in section["right"]] == [-1]
    assert [lane["id"] for lane in section.get("center", [])] == [0]


def test_lane_spec_splits_positive_and_negative_lane_numbers_with_lane_count():
    """When both sides have evidence, lanes are split across left/right lists."""

    sections = [{"s0": 0.0, "s1": 10.0}]

    lane_topology = {
        "lane_count": 4,
        "groups": {
            "A": ["A:1"],
            "B": ["B:2"],
            "C": ["C:-1"],
            "D": ["D:-2"],
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
            "B:2": {
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
            "C:-1": {
                "base_id": "C",
                "lane_no": -1,
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
            "D:-2": {
                "base_id": "D",
                "lane_no": -2,
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

    assert len(specs) == 1
    assert [lane["id"] for lane in specs[0]["left"]] == [1, 2]
    assert [lane["id"] for lane in specs[0]["right"]] == [-1, -2]


def test_lane_spec_handles_jpn_positive_only_topology():
    sections = [{"s0": 0.0, "s1": 30.0}]

    lane_topology = {
        "lane_count": 6,
        "groups": {
            "5.13001000000048e+18": ["lane::1"],
            "5.13001000000049e+18": ["lane::2"],
            "5.13001000000050e+18": ["lane::3"],
        },
        "lanes": {
            "lane::1": {
                "base_id": "5.13001000000048e+18",
                "lane_no": 1,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 30.0,
                        "width": 3.5,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {},
                    }
                ],
            },
            "lane::2": {
                "base_id": "5.13001000000049e+18",
                "lane_no": 2,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 30.0,
                        "width": 3.5,
                        "successors": [],
                        "predecessors": [],
                        "line_positions": {},
                    }
                ],
            },
            "lane::3": {
                "base_id": "5.13001000000050e+18",
                "lane_no": 3,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 30.0,
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

    assert len(specs) == 1
    left_ids = [lane["id"] for lane in specs[0]["left"]]

    assert left_ids == [1, 2, 3]
    assert specs[0]["right"] == []


def test_write_xodr_ignores_zero_length_centerline_segments(tmp_path):
    centerline = _SimpleCenterline({
        "s": [0.0, 5.0, 5.0, 10.0],
        "x": [0.0, 5.0, 5.0, 10.0],
        "y": [0.0, 0.0, 0.0, 0.0],
        "hdg": [0.0, 0.0, 0.0, 0.0],
    })

    sections = [{"s0": 0.0, "s1": 10.0}]
    lane_specs = build_lane_spec(
        sections,
        _simple_lane_topology(10.0),
        defaults={},
        lane_div_df=None,
    )

    out_file = Path(tmp_path) / "zero_length_center.xodr"
    write_xodr(centerline, sections, lane_specs, out_file)

    root = ET.parse(out_file).getroot()
    geometries = root.findall(".//planView/geometry")
    lengths = [float(elem.attrib["length"]) for elem in geometries]

    assert len(lengths) == 2
    assert all(length > 0 for length in lengths)
    assert pytest.approx(float(geometries[1].attrib["s"])) == 5.0


def test_write_xodr_skips_zero_length_geometry_segments(tmp_path):
    sections = [{"s0": 0.0, "s1": 10.0}]
    lane_specs = build_lane_spec(
        sections,
        _simple_lane_topology(10.0),
        defaults={},
        lane_div_df=None,
    )

    segments = [
        {"s": 0.0, "x": 0.0, "y": 0.0, "hdg": 0.0, "length": 5.0, "curvature": 0.0},
        {"s": 5.0, "x": 5.0, "y": 0.0, "hdg": 0.0, "length": 0.0, "curvature": 0.0},
        {"s": 5.0, "x": 5.0, "y": 0.0, "hdg": 0.0, "length": 5.0, "curvature": 0.01},
    ]

    out_file = Path(tmp_path) / "zero_length_geometry.xodr"
    write_xodr(_make_centerline(), sections, lane_specs, out_file, geometry_segments=segments)

    root = ET.parse(out_file).getroot()
    geometries = root.findall(".//planView/geometry")

    assert len(geometries) == 2
    assert pytest.approx(float(geometries[1].attrib["s"])) == 5.0

    second = geometries[1]
    assert second.find("arc") is not None, "non-zero curvature segment should render as arc"
