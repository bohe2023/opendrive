import pytest

from csv2xodr.normalize.us_adapters import merge_lane_width_into_links
from csv2xodr.simpletable import DataFrame


def test_merge_lane_width_into_links_injects_width_column():
    lane_link = DataFrame(
        [
            {"Lane Number": "15", "Offset[cm]": "0", "End Offset[cm]": "100"},
            {"Lane Number": "1", "Offset[cm]": "0", "End Offset[cm]": "100"},
        ],
        columns=["Lane Number", "Offset[cm]", "End Offset[cm]"],
    )

    lane_width = DataFrame(
        [
            {"Lane Number": "15", "幅員値[m]": "3.5"},
            {"Lane Number": "15", "幅員値[m]": "3.7"},
            {"Lane Number": "1", "幅員値[m]": "3.0"},
        ],
        columns=["Lane Number", "幅員値[m]"],
    )

    enriched = merge_lane_width_into_links(lane_link, lane_width)
    assert enriched is not None
    assert "幅員" in enriched.columns

    widths = enriched["幅員"].to_list()
    assert pytest.approx(widths[0], rel=1e-6) == 360.0
    assert pytest.approx(widths[1], rel=1e-6) == 300.0
