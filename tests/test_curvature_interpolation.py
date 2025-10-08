from __future__ import annotations

from pathlib import Path
import sys

import csv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pythonProject.interpolate_curvature import (
    SHAPE_INDEX_COLUMN,
    interpolate_group,
    interpolate_shape_indices,
    process_file,
)


def _make_row(
    *,
    lane_id: str,
    index: int,
    lane_count: str = "2",
    heading: str | None = None,
    curvature: str | None = None,
) -> dict[str, str]:
    return {
        "logTime": lane_id,
        "Instance ID": lane_id,
        "Is Retransmission": "False",
        "Path Id": lane_id,
        "Offset[cm]": lane_id,
        "End Offset[cm]": lane_id,
        "Lane Number": lane_id,
        "曲率情報のレーン数": lane_count,
        SHAPE_INDEX_COLUMN: str(index),
        "方位角[deg]": heading or f"{100 + index:.1f}",
        "曲率値[rad/m]": curvature or f"{index / 10:.4f}",
        "精度情報": "65535",
    }


def test_interpolate_group_fills_internal_gaps_with_previous_values():
    rows = [
        _make_row(lane_id="A", index=0, heading="10.0", curvature="0.1000"),
        _make_row(lane_id="A", index=2, heading="30.0", curvature="0.3000"),
    ]

    result = interpolate_group(rows)
    indices = [int(row[SHAPE_INDEX_COLUMN]) for row in result]

    assert indices == [0, 1, 2]
    assert result[1]["方位角[deg]"] == "10.0"
    assert result[1]["曲率値[rad/m]"] == "0.1000"


def test_interpolate_group_uses_following_rows_when_initial_index_missing():
    rows = [
        _make_row(lane_id="B", index=1, heading="20.0", curvature="0.2000"),
        _make_row(lane_id="B", index=1, heading="21.0", curvature="0.2100"),
    ]

    result = interpolate_group(rows)
    indices = [int(row[SHAPE_INDEX_COLUMN]) for row in result]

    assert indices == [0, 0, 1, 1]
    assert result[0]["方位角[deg]"] == "20.0"
    assert result[1]["方位角[deg]"] == "21.0"


def test_interpolate_shape_indices_resets_between_lanes():
    lane_a = [
        _make_row(lane_id="A", index=0, heading="15.0"),
        _make_row(lane_id="A", index=1, heading="25.0"),
    ]
    lane_b = [
        _make_row(lane_id="B", index=2, heading="35.0"),
    ]

    result = interpolate_shape_indices(lane_a + lane_b)

    a_indices = [int(row[SHAPE_INDEX_COLUMN]) for row in result[:2]]
    b_indices = [int(row[SHAPE_INDEX_COLUMN]) for row in result[2:]]

    assert a_indices == [0, 1]
    assert b_indices == [0, 1, 2]
    assert result[2]["方位角[deg]"] == "35.0"


def test_process_file_updates_csv(tmp_path: Path):
    source = tmp_path / "curvature.csv"
    rows = [
        _make_row(lane_id="C", index=1, heading="11.0"),
        _make_row(lane_id="C", index=2, heading="12.0"),
    ]

    fieldnames = list(rows[0].keys())
    with source.open("w", encoding="cp932", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    added = process_file(source, encoding="cp932")

    with source.open("r", encoding="cp932", newline="") as handle:
        reader = csv.DictReader(handle)
        updated = list(reader)

    assert added == 1
    assert [int(row[SHAPE_INDEX_COLUMN]) for row in updated] == [0, 1, 2]
    assert updated[0]["方位角[deg]"] == "11.0"
