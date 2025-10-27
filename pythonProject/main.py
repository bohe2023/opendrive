"""CSV変換フローを実行する簡易CLIユーティリティ。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from csv2xodr.csv2xodr import convert_dataset  # noqa: E402
from pythonProject import add_shape_index, interpolate_curvature, postprocess_xodr  # noqa: E402


@dataclass(frozen=True)
class FormatPipeline:
    """各フォーマットに対応する処理パイプラインの定義。"""

    name: str
    input_dir: Path
    output_dir: Path
    config_path: Path
    output_filename: str
    runner: Callable[[str, str, str], Dict]


def build_pipeline_registry() -> Dict[str, FormatPipeline]:
    """フォーマット識別子とパイプラインの対応表を構築する。"""

    input_root = ROOT / "input_csv"
    output_root = ROOT / "out"
    config_root = ROOT / "csv2xodr"

    return {
        "JPN": FormatPipeline(
            name="JPN",
            input_dir=input_root / "JPN",
            output_dir=output_root / "JPN",
            config_path=config_root / "config.yaml",
            output_filename="map.xodr",
            runner=convert_dataset,
        ),
        "US": FormatPipeline(
            name="US",
            input_dir=input_root / "US",
            output_dir=output_root / "US",
            config_path=config_root / "config_us.yaml",
            output_filename="map.xodr",
            runner=_run_us_pipeline,
        ),
    }


def _run_us_pipeline(input_dir: str, output_path: str, config_path: str) -> Dict:
    """米国向けデータの変換を実行する薄いラッパー。"""

    return convert_dataset(input_dir, output_path, config_path)


def run_pipeline(pipeline: FormatPipeline) -> Dict:
    """指定パイプラインを実行して統計情報を返す。"""

    if not pipeline.input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {pipeline.input_dir}")

    # 出力ディレクトリを整備してから処理を開始する
    pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = pipeline.output_dir / pipeline.output_filename
    return pipeline.runner(str(pipeline.input_dir), str(output_path), str(pipeline.config_path))


def preprocess_csv_sources(root: Path) -> None:
    """OpenDRIVE生成前にCSV側の前処理を行う。"""

    print("CSV前処理フローを開始します……")

    print(" 1/2: レーン幾何の形状インデックスを更新中……")
    for path in add_shape_index.default_files(root):
        if not path.exists():
            print(f"    [SKIP] ファイルが見つかりません: {path}")
            continue

        # 元データへ形状インデックス列を付加
        add_shape_index.add_shape_index_column(path, encoding=add_shape_index.DEFAULT_ENCODING)
        print(f"    [OK] 形状インデックス列を更新しました: {path}")

    print(" 2/2: 曲率CSVの欠損インデックスを補間中……")
    for path in interpolate_curvature.default_files(root):
        if not path.exists():
            print(f"    [SKIP] ファイルが見つかりません: {path}")
            continue

        added_rows = interpolate_curvature.process_file(
            path, encoding=interpolate_curvature.DEFAULT_ENCODING
        )
        if added_rows:
            print(f"    [OK] {path}: {added_rows} 行の補間データを追加しました")
        else:
            print(f"    [OK] {path}: 補間は不要でした")


def iter_targets(
    registry: Dict[str, FormatPipeline], selected_format: Optional[str], convert_all: bool
) -> Iterable[FormatPipeline]:
    """CLI引数に応じて実行対象となるパイプラインを返す。"""

    if convert_all:
        return registry.values()
    if selected_format is not None:
        return (registry[selected_format],)
    # 明示指定がなければ日規（JPN）のみ処理する
    return (registry["JPN"],)


def main(argv: Optional[Iterable[str]] = None) -> None:
    registry = build_pipeline_registry()
    parser = argparse.ArgumentParser(description="CSV仕様に基づきOpenDRIVEへ変換します。")
    parser.add_argument(
        "--format",
        choices=sorted(registry.keys()),
        help="変換対象のCSVフォーマット（省略時はJPN）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="対応する全フォーマットを一括変換",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.all:
        preprocess_csv_sources(ROOT)

    for pipeline in iter_targets(registry, args.format, args.all):
        print(f"{pipeline.name} フォーマットの変換を開始します……")
        try:
            stats = run_pipeline(pipeline)
        except NotImplementedError as exc:
            print(f"[{pipeline.name}] 未対応の機能です: {exc}")
            continue
        except FileNotFoundError as exc:
            print(f"[{pipeline.name}] 入力ディレクトリが不足しています: {exc}")
            continue

        print(json.dumps(stats, ensure_ascii=False, indent=2))

        xodr_path = stats.get("xodr_file", {}).get("path")
        if xodr_path:
            postprocess_xodr.patch_file(Path(xodr_path), verbose=True)


if __name__ == "__main__":  # pragma: no cover - CLI のエントリーポイント
    main()
