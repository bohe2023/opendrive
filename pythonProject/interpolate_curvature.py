"""Utilities for interpolating missing curvature shape indices in CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

SHAPE_INDEX_COLUMN = "形状インデックス"
LANE_COUNT_COLUMN = "曲率情報のレーン数"
DEFAULT_ENCODING = "cp932"

# Columns that uniquely describe a lane segment.  Rows sharing the same values for
# these fields are considered part of the same interpolation group.
GROUP_KEY_COLUMNS: Sequence[str] = (
    "logTime",
    "Instance ID",
    "Is Retransmission",
    "Path Id",
    "Offset[cm]",
    "End Offset[cm]",
    "Lane Number",
    LANE_COUNT_COLUMN,
)


def _parse_int(value: str) -> int:
    """Best-effort conversion of ``value`` to ``int``."""

    value = (value or "").strip()
    if not value:
        raise ValueError("cannot parse empty value as int")
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"invalid integer value: {value!r}") from exc


def _clone_with_index(row: Mapping[str, str], index: int) -> MutableMapping[str, str]:
    clone = dict(row)
    clone[SHAPE_INDEX_COLUMN] = str(index)
    return clone


def _group_rows(rows: Iterable[MutableMapping[str, str]]) -> Iterable[List[MutableMapping[str, str]]]:
    """Yield consecutive groups of rows sharing the same lane descriptor."""

    current_key: List[str] | None = None
    current_group: List[MutableMapping[str, str]] = []

    for row in rows:
        key = [row.get(column, "") for column in GROUP_KEY_COLUMNS]
        if current_key is None:
            current_key = key
        if key != current_key:
            yield current_group
            current_group = [row]
            current_key = key
        else:
            current_group.append(row)

    if current_group:
        yield current_group


def interpolate_group(rows: Sequence[MutableMapping[str, str]]) -> List[MutableMapping[str, str]]:
    """Fill missing shape indices within ``rows`` belonging to the same lane."""

    interpolated: List[MutableMapping[str, str]] = []
    last_template: MutableMapping[str, str] | None = None
    expected_index = 0

    for row in rows:
        current_index = _parse_int(row[SHAPE_INDEX_COLUMN])

        if interpolated and current_index < expected_index:
            # Index reset (e.g. next measurement block).  Start a fresh sequence
            # without attempting to back-fill values from the previous block.
            cloned = dict(row)
            interpolated.append(cloned)
            last_template = dict(cloned)
            expected_index = current_index + 1
            continue

        while current_index > expected_index:
            template = last_template if last_template is not None else dict(row)
            clone = _clone_with_index(template, expected_index)
            interpolated.append(clone)
            last_template = dict(clone)
            expected_index += 1

        cloned = dict(row)
        interpolated.append(cloned)
        last_template = dict(cloned)
        expected_index = current_index + 1

    return interpolated


def interpolate_shape_indices(rows: Iterable[MutableMapping[str, str]]) -> List[MutableMapping[str, str]]:
    """Interpolate missing shape indices across all lane groups."""

    output: List[MutableMapping[str, str]] = []
    for group in _group_rows(list(rows)):
        output.extend(interpolate_group(group))
    return output


def process_file(path: Path, *, encoding: str = DEFAULT_ENCODING) -> int:
    """Interpolate missing shape indices for ``path`` in place.

    Returns the number of rows that were added during the interpolation.
    """

    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    updated_rows = interpolate_shape_indices(rows)
    added_rows = len(updated_rows) - len(rows)

    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    return added_rows


def _default_files(root: Path) -> List[Path]:
    return [
        root / "input_csv" / "JPN" / "PROFILETYPE_MPU_ZGM_CURVATURE.csv",
        root / "input_csv" / "US" / "PROFILETYPE_MPU_US_CURVATURE.csv",
    ]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate missing CURVATURE.csv shape indices.")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="CSV files to update. Defaults to the standard JPN/US curvature datasets.",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="Character encoding used to read/write the CSV files (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    files = args.files or _default_files(Path(__file__).resolve().parents[1])

    for path in files:
        if not path.exists():
            print(f"[SKIP] {path} (not found)")
            continue

        added = process_file(path, encoding=args.encoding)
        if added:
            print(f"[OK] {path}: added {added} interpolated row(s)")
        else:
            print(f"[OK] {path}: no missing shape indices detected")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
