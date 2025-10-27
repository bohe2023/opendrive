"""車線幾何CSVへ形状インデックス列を付与する補助モジュール。"""

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
    """単一路線におけるインデックス状態を保持する内部構造体。"""

    next_index: int = 0
    point_count: Optional[int] = None


def _parse_int(value: str) -> Optional[int]:
    """可能であれば ``value`` を ``int`` へ変換する。"""

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
    """車線単位で0始まりの形状インデックスを割り当てる。"""

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

        # 形状インデックス列へ書き戻す
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
    parser = argparse.ArgumentParser(description="レーン幾何CSVへ形状インデックス列を追加します。")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="更新対象のCSVファイル（省略時は標準のJPN/USデータセット）",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="CSVの入出力に使用する文字コード（既定値: %(default)s）",
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
            print(f"[SKIP] ファイルが見つかりません: {path}")
            continue

        add_shape_index_column(path, encoding=args.encoding)
        print(f"[OK] 形状インデックス列を付与しました: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
