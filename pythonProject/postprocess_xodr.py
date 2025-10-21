"""Post-processing helpers for generated OpenDRIVE files.

This module currently focuses on ensuring that each ``laneSection``
has a usable centre lane definition.  MATLAB 的 Driving Scenario Designer
在导入 OpenDRIVE 时会将中心车道作为参考线向两侧外推。当中心车道
缺失 ``<width>`` 元素或被标记为 ``type="none"`` 时，导入器无法推导
截面宽度，进而报错。

为了提高兼容性，我们在导出的 ``.xodr`` 中遍历每个 ``laneSection``，
若发现中心车道缺少宽度定义，就补上一段零宽度曲线，并在需要时
将其 ``type`` 属性强制设置为 ``driving``。这样即使原始数据未提供
中心车道宽度，导入器也能够识别到有效的截面信息。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_WIDTH_ATTRS = {
    "sOffset": "0",
    "a": "0",
    "b": "0",
    "c": "0",
    "d": "0",
}


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
    """Ensure every centre lane has an explicit ``<width>`` child.

    Parameters
    ----------
    path:
        Path to the ``.xodr`` file that should be patched in-place.
    force_lane_type:
        Optional lane ``type`` that should be enforced whenever a centre lane
        is missing width information.  If ``None`` the original ``type`` is
        preserved.
    width_attrs:
        Attribute dictionary used to populate the injected ``<width>`` nodes.

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

        has_width = any(child.tag == "width" for child in lane)
        if not has_width:
            attrs = dict(width_attrs or DEFAULT_WIDTH_ATTRS)
            width_element = ET.Element("width", attrs)

            link = lane.find("link")
            if link is not None:
                index = list(lane).index(link) + 1
                lane.insert(index, width_element)
            else:
                lane.append(width_element)

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
        changed = ensure_center_lane_width(path, force_lane_type="driving")
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

    parser = argparse.ArgumentParser(description="为中心车道补写宽度段")
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

