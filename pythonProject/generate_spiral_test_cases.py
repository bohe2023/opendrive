"""Generate minimal OpenDRIVE files for spiral comparison tests.

该脚本会在指定的输出目录下生成两个 xodr 文件：

1. `spiral_zero_diff.xodr`：包含曲率差为 0 的螺旋段。
2. `spiral_nonzero_diff.xodr`：包含曲率差不为 0 的螺旋段。

脚本旨在帮助验证 MATLAB Driving Scenario Toolbox 在处理
零曲率差螺旋段时的行为差异。
"""

from __future__ import annotations

import argparse
import os
import textwrap
import xml.etree.ElementTree as ET


HEADER_ATTRS = {
    "revMajor": "1",
    "revMinor": "4",
    "name": "spiral_test",
    "version": "1.00",
    "date": "2024-01-01T00:00:00",
    "north": "0",
    "south": "0",
    "east": "0",
    "west": "0",
}


def _build_lane_section(parent: ET.Element, length: float) -> None:
    lanes = ET.SubElement(parent, "lanes")
    lane_section = ET.SubElement(lanes, "laneSection", attrib={"s": "0", "singleSide": "false"})

    center = ET.SubElement(lane_section, "center")
    ET.SubElement(
        center,
        "lane",
        attrib={"id": "0", "type": "none", "level": "false"},
    )

    left = ET.SubElement(lane_section, "left")
    driving_lane = ET.SubElement(
        left,
        "lane",
        attrib={"id": "1", "type": "driving", "level": "false"},
    )
    ET.SubElement(
        driving_lane,
        "width",
        attrib={"sOffset": "0", "a": "3.5", "b": "0", "c": "0", "d": "0"},
    )

    ET.SubElement(parent, "elevationProfile")
    ET.SubElement(parent, "lateralProfile")


def _build_plan_view(parent: ET.Element, length: float, curvature_start: float, curvature_end: float) -> None:
    plan_view = ET.SubElement(parent, "planView")
    geometry = ET.SubElement(
        plan_view,
        "geometry",
        attrib={
            "s": "0",
            "x": "0",
            "y": "0",
            "hdg": "0",
            "length": f"{length:.3f}",
        },
    )
    ET.SubElement(
        geometry,
        "spiral",
        attrib={
            "curvatureStart": f"{curvature_start:.12f}",
            "curvatureEnd": f"{curvature_end:.12f}",
        },
    )


def build_spiral_xodr(length: float, curvature_start: float, curvature_end: float) -> ET.Element:
    """构造只包含单条道路的最小 OpenDRIVE 结构。"""

    root = ET.Element("OpenDRIVE")
    header = ET.SubElement(root, "header", attrib=HEADER_ATTRS | {"length": f"{length:.3f}"})
    header.text = "\n"

    road = ET.SubElement(
        root,
        "road",
        attrib={
            "name": "spiral_test",
            "length": f"{length:.3f}",
            "id": "1",
            "junction": "-1",
        },
    )
    ET.SubElement(road, "type", attrib={"s": "0", "type": "driving"})
    _build_plan_view(road, length, curvature_start, curvature_end)
    _build_lane_section(road, length)

    return root


def save_tree(tree: ET.Element, path: str) -> None:
    ET.indent(tree, space="  ")
    xml_bytes = ET.tostring(tree, encoding="utf-8")
    with open(path, "wb") as f:
        f.write(b"<?xml version=\"1.0\" standalone=\"yes\"?>\n")
        f.write(xml_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成用于比较的最小螺旋 OpenDRIVE 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """示例：
            python generate_spiral_test_cases.py --output-dir out/spiral_tests
            """
        ),
    )
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--length", type=float, default=50.0, help="道路长度，单位米")
    parser.add_argument(
        "--curvature",
        type=float,
        default=0.001,
        help="非零螺旋段的曲率差（终止曲率 = 初始曲率 + 该值）",
    )
    parser.add_argument(
        "--start-curvature",
        type=float,
        default=0.0,
        help="螺旋段的起始曲率",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 零曲率差螺旋
    zero_tree = build_spiral_xodr(args.length, args.start_curvature, args.start_curvature)
    zero_path = os.path.join(args.output_dir, "spiral_zero_diff.xodr")
    save_tree(zero_tree, zero_path)

    # 非零曲率差螺旋
    nonzero_tree = build_spiral_xodr(
        args.length,
        args.start_curvature,
        args.start_curvature + args.curvature,
    )
    nonzero_path = os.path.join(args.output_dir, "spiral_nonzero_diff.xodr")
    save_tree(nonzero_tree, nonzero_path)

    print(f"已生成：{zero_path}")
    print(f"已生成：{nonzero_path}")


if __name__ == "__main__":
    main()
