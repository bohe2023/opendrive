from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from csv2xodr.normalize.core import build_elevation_profile
from csv2xodr.simpletable import DataFrame


def test_build_elevation_profile_averages_and_slopes():
    rows = [
        {
            "Offset[cm]": "0",
            "高さ[m]": "10.0",
            "Path Id": "1",
            "Is Retransmission": "False",
        },
        {
            "Offset[cm]": "100",
            "高さ[m]": "12.0",
            "Path Id": "1",
            "Is Retransmission": "False",
        },
        {
            "Offset[cm]": "100",
            "高さ[m]": "14.0",
            "Path Id": "1",
            "Is Retransmission": "False",
        },
        {
            "Offset[cm]": "200",
            "高さ[m]": "20.0",
            "Path Id": "2",
            "Is Retransmission": "False",
        },
        {
            "Offset[cm]": "200",
            "高さ[m]": "15.0",
            "Path Id": "1",
            "Is Retransmission": "True",
        },
    ]
    df = DataFrame(rows)

    profile = build_elevation_profile(df)

    assert len(profile) == 2
    first, second = profile

    assert first["s"] == 0.0
    assert first["a"] == 10.0
    assert first["b"] == 3.0  # (13 - 10) / (1 - 0)

    assert second["s"] == 1.0
    assert second["a"] == 13.0  # average of 12 and 14
    assert second["b"] == 3.0  # inherits slope from previous segment
