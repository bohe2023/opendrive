from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csv2xodr.simpletable import DataFrame
from csv2xodr.topology.core import build_lane_topology


def test_build_lane_topology_prefers_true_retransmission_segments():
    df = DataFrame(
        [
            {
                "Offset[cm]": "0",
                "End Offset[cm]": "100",
                "レーンID": "A",
                "レーン番号": "1",
                "Is Retransmission": "false",
            },
            {
                "Offset[cm]": "0",
                "End Offset[cm]": "100",
                "レーンID": "A",
                "レーン番号": "1",
                "Is Retransmission": "true",
            },
        ]
    )

    topology = build_lane_topology(df)

    assert "A:1" in topology["lanes"]
    segments = topology["lanes"]["A:1"]["segments"]
    assert len(segments) == 1
    assert segments[0]["is_retrans"] is True
