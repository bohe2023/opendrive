"""Post-processing helpers for generated OpenDRIVE files.

本模块用于统一每个 ``laneSection`` 的中心车道定义，使其满足 MATLAB
Driving Scenario Designer 对 OpenDRIVE 的导入要求。根据最新的兼容性
测试，中心车道应被标记为 ``type="none"`` 且不携带 ``<width>``
子节点，以便解析器仅以其作为参考线。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional


def _iter_center_lanes(tree: ET.ElementTree) -> Iterable[ET.Element]:
    """Yield all ``<lane>`` elements that belong to a ``<center>`` block."""

    root = tree.getroot()
    for lane_section in root.findall(".//laneSection"):
        center = lane_section.find("center")
        if center is None:
            continue
        for lane in center.findall("lane"):
            yield lane


def ensure_center_lane_width(
    path: Path,
    *,
    force_lane_type: Optional[str] = None,
    width_attrs: Optional[dict[str, str]] = None,
) -> bool:
    """Normalize centre lane ``type`` and remove obsolete ``<width>`` nodes.

    Parameters
    ----------
    path:
        Path to the ``.xodr`` file that should be patched in-place.
    force_lane_type:
        Optional lane ``type`` that should be enforced for each centre lane.
        If ``None`` the original ``type`` is preserved.
    width_attrs:
        Deprecated and ignored.  Present for backwards compatibility with
        earlier automation scripts.

    Returns
    -------
    bool
        ``True`` if the file was modified, ``False`` if no changes were
        necessary.
    """

    tree = ET.parse(path)
    updated = False

    for lane in _iter_center_lanes(tree):
        lane_updated = False

        if force_lane_type is not None and lane.get("type") != force_lane_type:
            lane.set("type", force_lane_type)
            lane_updated = True

        # Remove any width definitions so the lane acts purely as a reference.
        width_elements = [child for child in lane if child.tag == "width"]
        if width_elements:
            for element in width_elements:
                lane.remove(element)
            lane_updated = True

        if lane_updated:
            updated = True

    if not updated:
        return False

    tree.write(path, encoding="utf-8", xml_declaration=True)
    return True


def patch_file(path: Path, *, verbose: bool = True) -> bool:
    """Public helper used by CLI/automation hooks to patch a single file."""

    try:
        changed = ensure_center_lane_width(path, force_lane_type="none")
    except FileNotFoundError:
        if verbose:
            print(f"[SKIP] 未找到XODR文件: {path}")
        return False

    if changed and verbose:
        print(f"[OK] 已更新中心车道定义: {path}")
    elif verbose:
        print(f"[OK] 中心车道定义已满足要求: {path}")
    return changed


def patch_paths(paths: Iterable[Path], *, verbose: bool = True) -> int:
    """Patch multiple files and return the count of modified entries."""

    touched = 0
    for path in paths:
        if patch_file(path, verbose=verbose):
            touched += 1
    return touched


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Minimal CLI so the helper can be invoked manually if required."""

    import argparse

    parser = argparse.ArgumentParser(description="规范中心车道的类型与宽度定义")
    parser.add_argument("paths", nargs="+", help="需要修复的XODR文件路径")
    parser.add_argument(
        "--quiet", action="store_true", help="静默模式，仅返回退出码"
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    path_objs = [Path(p) for p in args.paths]

    modified = patch_paths(path_objs, verbose=not args.quiet)
    raise SystemExit(0 if modified >= 0 else 1)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    main()

