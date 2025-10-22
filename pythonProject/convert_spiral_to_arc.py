"""Convert all spiral segments in an OpenDRIVE file to constant-curvature arcs.

该脚本用于将 ``.xodr`` 路径中的所有 ``<spiral>`` 几何段替换为 ``<arc>``
段。替换时会读取 ``curvatureStart`` 与 ``curvatureEnd`` 属性，计算其
平均值作为 ``<arc>`` 的 ``curvature``，以达到消除螺旋段的目的。

Usage
-----
    python convert_spiral_to_arc.py /path/to/map.xodr

参数既可以是单个文件，也可以是目录。传入目录时脚本会在其下递归
查找所有 ``.xodr`` 文件并进行处理。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Iterable


def _iter_xodr_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield unique ``.xodr`` files from ``paths``.

    Directories are traversed recursively while plain files are yielded as-is.
    """

    seen: set[Path] = set()

    for path in paths:
        if path.is_dir():
            for file_path in path.rglob("*.xodr"):
                resolved = file_path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved
        else:
            resolved = path.resolve()
            if resolved.suffix.lower() != ".xodr":
                continue
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def _format_curvature(value: float) -> str:
    """Return a curvature string without introducing scientific notation."""

    # ``:.12g`` keeps the original precision reasonably intact while avoiding
    # trailing zeros. ``repr`` would introduce scientific notation for small
    # values which is less desirable for human inspection of XODR files.
    return f"{value:.12g}"


def convert_spiral_to_arc(path: Path) -> int:
    """Replace all ``<spiral>`` geometries with ``<arc>`` nodes.

    Parameters
    ----------
    path:
        The OpenDRIVE file to be patched in-place.

    Returns
    -------
    int
        Number of spiral elements that have been replaced.
    """

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:  # pragma: no cover - defensive branch
        print(f"[ERROR] 无法解析XODR文件: {path}: {exc}", file=sys.stderr)
        return 0

    root = tree.getroot()
    replaced = 0

    for geometry in root.findall(".//geometry"):
        children = list(geometry)
        for index, element in enumerate(children):
            if element.tag != "spiral":
                continue

            curv_start = element.get("curvatureStart")
            curv_end = element.get("curvatureEnd")
            if curv_start is None or curv_end is None:
                print(
                    f"[WARN] 缺少曲率信息，跳过: {path} (s={geometry.get('s', 'N/A')})",
                    file=sys.stderr,
                )
                continue

            try:
                average = (float(curv_start) + float(curv_end)) / 2.0
            except ValueError:
                print(
                    f"[WARN] 曲率值非数字，跳过: {path} (s={geometry.get('s', 'N/A')})",
                    file=sys.stderr,
                )
                continue

            arc = ET.Element("arc")
            arc.set("curvature", _format_curvature(average))

            # Replace the spiral element while preserving the order of siblings.
            geometry.remove(element)
            geometry.insert(index, arc)
            replaced += 1

    if replaced:
        tree.write(path, encoding="utf-8", xml_declaration=True)

    return replaced


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="将OpenDRIVE文件中的螺旋段替换为圆弧段")
    parser.add_argument(
        "paths",
        nargs="+",
        help="需要处理的 .xodr 文件或目录路径，可传入多个",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，仅在发生修改时输出",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    target_paths = [Path(p) for p in args.paths]

    total_replaced = 0
    for file_path in _iter_xodr_files(target_paths):
        replaced = convert_spiral_to_arc(file_path)
        if not args.quiet or replaced:
            status = "OK" if replaced else "SKIP"
            print(f"[{status}] {file_path} -> 替换 {replaced} 个螺旋段")
        total_replaced += replaced

    raise SystemExit(0 if total_replaced >= 0 else 1)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    main()
