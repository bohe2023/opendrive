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

    lane_specs = build_lane_spec(sections, lane_topo={}, defaults={}, lane_div_df=None)

    assert lane_specs[0]["predecessor"] is False
    assert lane_specs[0]["successor"] is True
    assert lane_specs[1]["predecessor"] is True
    assert lane_specs[1]["successor"] is False

    out_file = Path(tmp_path) / "lane_links.xodr"
    write_xodr(_make_centerline(), sections, lane_specs, out_file)

    root = ET.parse(out_file).getroot()
    lane_sections = root.find(".//lanes").findall("laneSection")

    first_left_link = lane_sections[0].find("./left/lane/link")
    assert first_left_link is not None
    assert first_left_link.find("predecessor") is None
    assert first_left_link.find("successor") is not None

    last_right_link = lane_sections[-1].find("./right/lane/link")
    assert last_right_link is not None
    assert last_right_link.find("predecessor") is not None
    assert last_right_link.find("successor") is None
