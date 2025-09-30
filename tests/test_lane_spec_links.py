import sys
import xml.etree.ElementTree as ET
from pathlib import Path

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


def test_lane_spec_uses_lane_count_when_only_positive_lane_numbers():
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

    specs = build_lane_spec(sections, lane_topology, defaults={}, lane_div_df=None)

    assert len(specs) == 1
    left_ids = [lane["id"] for lane in specs[0]["left"]]
    right_ids = [lane["id"] for lane in specs[0]["right"]]

    assert left_ids == [1, 2]
    assert right_ids == [-1, -2]
