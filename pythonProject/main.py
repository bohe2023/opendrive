"""Command line helper to convert CSV datasets into OpenDRIVE files.

This script understands different CSV layouts (currently JPN and US) and
dispatches them to their dedicated conversion pipeline.  Both formats reuse
the conversion machinery in ``csv2xodr.csv2xodr`` while supplying their
format-specific configuration files.
"""

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
    """Description of a conversion pipeline for a specific CSV flavour."""

    name: str
    input_dir: Path
    output_dir: Path
    config_path: Path
    output_filename: str
    runner: Callable[[str, str, str], Dict]


def build_pipeline_registry() -> Dict[str, FormatPipeline]:
    """Create the registry that maps format identifiers to pipelines."""

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
    """Execute the US-specific CSV → OpenDRIVE workflow."""

    return convert_dataset(input_dir, output_path, config_path)


def run_pipeline(pipeline: FormatPipeline) -> Dict:
    """Execute a pipeline and return the conversion statistics."""

    if not pipeline.input_dir.exists():
        raise FileNotFoundError(f"未找到输入目录: {pipeline.input_dir}")

    pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = pipeline.output_dir / pipeline.output_filename
    return pipeline.runner(str(pipeline.input_dir), str(output_path), str(pipeline.config_path))


def preprocess_csv_sources(root: Path) -> None:
    """Run the CSV preprocessing steps prior to OpenDRIVE conversion."""

    print("开始执行CSV预处理流程……")

    print(" 1/2: 更新车道几何形状索引……")
    for path in add_shape_index.default_files(root):
        if not path.exists():
            print(f"    [SKIP] 未找到文件: {path}")
            continue

        add_shape_index.add_shape_index_column(path, encoding=add_shape_index.DEFAULT_ENCODING)
        print(f"    [OK] 已更新形状索引列: {path}")

    print(" 2/2: 插值曲率缺失的形状索引……")
    for path in interpolate_curvature.default_files(root):
        if not path.exists():
            print(f"    [SKIP] 未找到文件: {path}")
            continue

        added_rows = interpolate_curvature.process_file(
            path, encoding=interpolate_curvature.DEFAULT_ENCODING
        )
        if added_rows:
            print(f"    [OK] {path}: 新增 {added_rows} 行插值数据")
        else:
            print(f"    [OK] {path}: 无需插值，保持原状")


def iter_targets(
    registry: Dict[str, FormatPipeline], selected_format: Optional[str], convert_all: bool
) -> Iterable[FormatPipeline]:
    """Return the pipelines that should be executed based on CLI arguments."""

    if convert_all:
        return registry.values()
    if selected_format is not None:
        return (registry[selected_format],)
    # 默认仅处理日规（JPN）格式
    return (registry["JPN"],)


def main(argv: Optional[Iterable[str]] = None) -> None:
    registry = build_pipeline_registry()
    parser = argparse.ArgumentParser(description="根据CSV格式转换为OpenDRIVE。")
    parser.add_argument(
        "--format",
        choices=sorted(registry.keys()),
        help="需要转换的CSV格式标识（默认JPN）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="一次性转换所有支持的格式。",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.all:
        preprocess_csv_sources(ROOT)

    for pipeline in iter_targets(registry, args.format, args.all):
        print(f"开始处理 {pipeline.name} 格式……")
        try:
            stats = run_pipeline(pipeline)
        except NotImplementedError as exc:
            print(f"[{pipeline.name}] 暂未完成: {exc}")
            continue
        except FileNotFoundError as exc:
            print(f"[{pipeline.name}] 输入目录缺失: {exc}")
            continue

        print(json.dumps(stats, ensure_ascii=False, indent=2))

        xodr_path = stats.get("xodr_file", {}).get("path")
        if xodr_path:
            postprocess_xodr.patch_file(Path(xodr_path), verbose=True)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
