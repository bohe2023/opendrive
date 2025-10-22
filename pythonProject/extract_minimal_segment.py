"""从大型 OpenDRIVE 文件中提取最简单的道路片段。

默认行为：
    * 仅保留第一条道路（或通过 --road-id 指定的道路）。
    * 在该道路内，仅保留前 N 段几何（通过 --max-geometries 控制）。
    * 自动更新 Header 与 Road 的长度字段，方便手动排查。

脚本不会修改输入文件，而是生成一个新的裁剪结果。
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"无法解析浮点数：{value}") from exc


def _update_length(header: ET.Element, road: ET.Element) -> None:
    geometries = road.find("planView")
    if geometries is None:
        return

    lengths = []
    for geom in geometries.findall("geometry"):
        length_attr = geom.get("length")
        if length_attr is None:
            continue
        lengths.append(_float(length_attr))

    if not lengths:
        return

    total_length = math.fsum(lengths)
    header.set("length", f"{total_length:.3f}")
    road.set("length", f"{total_length:.3f}")


def extract_segment(tree: ET.ElementTree, road_id: str | None, max_geometries: int | None) -> ET.ElementTree:
    root = tree.getroot()
    roads = root.findall("road")
    if not roads:
        raise ValueError("未在 XODR 中找到任何 road 节点")

    selected_road = None
    if road_id is not None:
        for road in roads:
            if road.get("id") == road_id:
                selected_road = road
                break
        if selected_road is None:
            raise ValueError(f"未找到 id 为 {road_id} 的 road")
    else:
        selected_road = roads[0]

    # 仅保留选中道路
    for road in list(roads):
        if road is not selected_road:
            root.remove(road)

    plan_view = selected_road.find("planView")
    if plan_view is None:
        raise ValueError("选中的 road 不包含 planView")

    if max_geometries is not None:
        geometries = plan_view.findall("geometry")
        if len(geometries) > max_geometries:
            for geom in geometries[max_geometries:]:
                plan_view.remove(geom)

    header = root.find("header")
    if header is not None:
        _update_length(header, selected_road)

    return tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="裁剪 OpenDRIVE 文件，保留最简单片段")
    parser.add_argument("input", help="原始 xodr 文件路径")
    parser.add_argument("output", help="输出 xodr 文件路径")
    parser.add_argument("--road-id", help="需要保留的 road id，如果不指定则使用第一条", default=None)
    parser.add_argument(
        "--max-geometries",
        type=int,
        default=None,
        help="planView 中保留的 geometry 数量",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tree = ET.parse(args.input)
    tree = extract_segment(tree, args.road_id, args.max_geometries)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"已写入：{args.output}")


if __name__ == "__main__":
    main()
