"""Utility for adding a shape-index column to lane geometry CSV files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional


SHAPE_INDEX_COLUMN = "形状インデックス"
DEFAULT_ENCODING = "cp932"


@dataclass
class _LaneState:
    """Book-keeping data that tracks shape indices for a single lane."""

    next_index: int = 0
    point_count: Optional[int] = None


def _parse_int(value: str) -> Optional[int]:
    """Convert ``value`` to :class:`int` if possible."""

    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def assign_shape_indices(
    rows: Iterable[MutableMapping[str, str]],
    *,
    lane_id_column: str = "Lane ID",
    point_count_column: str = "形状要素点数",
    shape_index_column: str = SHAPE_INDEX_COLUMN,
) -> List[MutableMapping[str, str]]:
    """Assign a 0-based shape index to each row grouped by lane id.

    The function walks through ``rows`` in order and maintains a counter per
    lane identifier.  Whenever the counter reaches the reported number of
    geometry points (``形状要素点数``), it wraps back to zero.  This keeps the
    produced indices within the ``[0, 形状要素点数 - 1]`` range even for datasets
    that contain multiple transmissions of the same lane geometry.
    """

    states: Dict[str, _LaneState] = {}
    processed: List[MutableMapping[str, str]] = []

    for row in rows:
        lane_id = str(row.get(lane_id_column, "")).strip()
        if not lane_id:
            processed.append(row)
            continue

        state = states.setdefault(lane_id, _LaneState())
        point_count = _parse_int(str(row.get(point_count_column, "")))

        if point_count and point_count > 0:
            if state.point_count != point_count:
                state.next_index = 0
                state.point_count = point_count

            index_value = state.next_index
            state.next_index = (state.next_index + 1) % point_count
        else:
            index_value = state.next_index
            state.next_index += 1
            state.point_count = None

        row[shape_index_column] = str(index_value)
        processed.append(row)

    return processed
def add_shape_index_column(path: Path, *, encoding: str = DEFAULT_ENCODING) -> None:
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    assign_shape_indices(rows)

    if SHAPE_INDEX_COLUMN not in fieldnames:
        fieldnames.append(SHAPE_INDEX_COLUMN)

    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add shape index columns to lane geometry CSV files.")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="CSV files to update. Defaults to the standard JPN/US lane geometry datasets.",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="Character encoding used to read/write the CSV files (default: %(default)s).",
    )
    return parser.parse_args()
def default_files(root: Optional[Path] = None) -> List[Path]:
    base = root or Path(__file__).resolve().parents[1]
    return [
        base / "input_csv" / "JPN" / "LanesGeometryProfile.csv",
        base / "input_csv" / "US" / "LanesGeometryProfile_US.csv",
    ]
def main() -> None:
    args = _parse_arguments()
    files = args.files or default_files(Path(__file__).resolve().parents[1])

    for path in files:
        if not path.exists():
            print(f"[SKIP] {path} (not found)")
            continue

        add_shape_index_column(path, encoding=args.encoding)
        print(f"[OK] Added shape index column to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
