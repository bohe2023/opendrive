"""曲率CSVで欠損した形状インデックスを補完するためのツール群。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

SHAPE_INDEX_COLUMN = "形状インデックス"
LANE_COUNT_COLUMN = "曲率情報のレーン数"
DEFAULT_ENCODING = "cp932"

# レーン区間を一意に識別するカラム群。これらが一致する行は同一グループとみなす。
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

# CSVによりヘッダー名が揺れるケースがあるため、カラム名のエイリアスも保持する。
GROUP_KEY_ALIASES: Mapping[str, Sequence[str]] = {
    "logTime": ("logTime", "ExpTime"),
}


def _parse_int(value: str) -> int:
    """文字列を可能な限り ``int`` へ変換する。"""

    value = (value or "").strip()
    if not value:
        raise ValueError("空文字列を整数に変換できません")
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"不正な整数値です: {value!r}") from exc


def _clone_with_index(row: Mapping[str, str], index: int) -> MutableMapping[str, str]:
    clone = dict(row)
    clone[SHAPE_INDEX_COLUMN] = str(index)
    return clone


def _resolve_group_column(row: Mapping[str, str], column: str) -> str:
    """グループキー構築時に利用する値を取得する。"""

    candidates = GROUP_KEY_ALIASES.get(column, (column,))
    for name in candidates:
        value = row.get(name)
        if value not in (None, ""):
            return value

    # 該当カラムが無い場合は空文字で代替し、グループ化の決定性を保つ。
    return row.get(candidates[0], "") or ""


def _group_rows(rows: Iterable[MutableMapping[str, str]]) -> Iterable[List[MutableMapping[str, str]]]:

    current_key: List[str] | None = None
    current_group: List[MutableMapping[str, str]] = []

    for row in rows:
        key = [
            _resolve_group_column(row, column) for column in GROUP_KEY_COLUMNS
        ]
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

    interpolated: List[MutableMapping[str, str]] = []
    last_template: MutableMapping[str, str] | None = None
    expected_index = 0

    for row in rows:
        current_index = _parse_int(row[SHAPE_INDEX_COLUMN])

        if interpolated and current_index < expected_index:
            # インデックスが巻き戻った場合は新しい計測ブロックとして扱う。
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
    """全レーンで欠損インデックスを補完した行一覧を返す。"""

    output: List[MutableMapping[str, str]] = []
    for group in _group_rows(list(rows)):
        output.extend(interpolate_group(group))
    return output


def process_file(path: Path, *, encoding: str = DEFAULT_ENCODING) -> int:
    """対象ファイルの欠損インデックスを補完し追加行数を返す。"""

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


def default_files(root: Path | None = None) -> List[Path]:
    """標準的に処理すべき曲率CSVの一覧を返す。"""

    base = root or Path(__file__).resolve().parents[1]
    return [
        base / "input_csv" / "JPN" / "PROFILETYPE_MPU_ZGM_CURVATURE.csv",
        base / "input_csv" / "US" / "PROFILETYPE_MPU_US_CURVATURE.csv",
    ]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="曲率CSVの欠損形状インデックスを補間します。")
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


def main() -> None:
    args = _parse_arguments()
    files = args.files or default_files(Path(__file__).resolve().parents[1])

    for path in files:
        if not path.exists():
            print(f"[SKIP] ファイルが見つかりません: {path}")
            continue

        added = process_file(path, encoding=args.encoding)
        if added:
            print(f"[OK] {path}: 補間行を {added} 行追加しました")
        else:
            print(f"[OK] {path}: 欠損インデックスは検出されませんでした")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
