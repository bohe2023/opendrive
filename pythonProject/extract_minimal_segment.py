"""大型OpenDRIVEから検証に十分な最小道路片段を切り出すスクリプト。"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"浮動小数点数を解釈できません: {value}") from exc


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


def _limit_children(element: ET.Element, tag: str, max_count: int) -> None:
    children = element.findall(tag)
    for child in children[max_count:]:
        element.remove(child)


def _prune_lane_sections(lanes: ET.Element, max_lane_sections: int | None) -> None:
    lane_sections = lanes.findall("laneSection")
    if max_lane_sections is not None and len(lane_sections) > max_lane_sections:
        for section in lane_sections[max_lane_sections:]:
            lanes.remove(section)

    kept_sections = lanes.findall("laneSection")
    if not kept_sections:
        return

    last_index = len(kept_sections) - 1

    for index, section in enumerate(kept_sections):
        has_prev = index > 0
        has_next = index < last_index
        _cleanup_lane_links(section, has_prev=has_prev, has_next=has_next)
        _simplify_lane_section(section)

    lane_offsets = lanes.findall("laneOffset")
    if lane_offsets:
        # laneSection が1つだけの場合は最初の laneOffset のみ維持する。
        if len(kept_sections) == 1:
            for offset in lane_offsets[1:]:
                lanes.remove(offset)


def _cleanup_lane_links(section: ET.Element, *, has_prev: bool, has_next: bool) -> None:
    for side_name in ("left", "center", "right"):
        side = section.find(side_name)
        if side is None:
            continue

        for lane in side.findall("lane"):
            link = lane.find("link")
            if link is None:
                continue

            if not has_prev:
                predecessor = link.find("predecessor")
                if predecessor is not None:
                    link.remove(predecessor)

            if not has_next:
                successor = link.find("successor")
                if successor is not None:
                    link.remove(successor)

            if len(list(link)) == 0:
                lane.remove(link)


def _simplify_lane_section(section: ET.Element) -> None:
    for side_name in ("left", "center", "right"):
        side = section.find(side_name)
        if side is None:
            continue

        for lane in side.findall("lane"):
            _limit_children(lane, "width", 1)
            _limit_children(lane, "speed", 1)
            _limit_children(lane, "access", 1)
            _limit_children(lane, "height", 1)

            for road_mark in lane.findall("roadMark"):
                _limit_children(road_mark, "explicit", 1)
                explicit = road_mark.find("explicit")
                if explicit is not None:
                    _limit_children(explicit, "geometry", 1)

            _limit_children(lane, "roadMark", 1)


def _cleanup_road_links(road: ET.Element) -> None:
    link = road.find("link")
    if link is None:
        return

    for child in list(link):
        if child.tag in {"predecessor", "successor"}:
            link.remove(child)

    if len(list(link)) == 0:
        road.remove(link)


def _simplify_profiles(road: ET.Element) -> None:
    profile_specs: list[tuple[str, tuple[str, ...], bool]] = [
        ("elevationProfile", ("elevation",), False),
        ("lateralProfile", ("superelevation", "crossfall", "shape"), False),
        ("objects", ("object",), True),
        ("signals", ("signal",), True),
        ("surface", ("CRG", "sampledHeight"), True),
    ]

    for parent_tag, child_tags, remove_whole in profile_specs:
        parent = road.find(parent_tag)
        if parent is None:
            continue

        if remove_whole:
            road.remove(parent)
            continue

        for child_tag in child_tags:
            _limit_children(parent, child_tag, 1)


def extract_segment(
    tree: ET.ElementTree,
    road_id: str | None,
    max_geometries: int | None,
    max_lane_sections: int | None,
) -> ET.ElementTree:
    root = tree.getroot()
    roads = root.findall("road")
    if not roads:
        raise ValueError("XODR内にroadノードが見つかりません")

    selected_road = None
    if road_id is not None:
        for road in roads:
            if road.get("id") == road_id:
                selected_road = road
                break
        if selected_road is None:
            raise ValueError(f"IDが {road_id} のroadが見つかりません")
    else:
        selected_road = roads[0]

    # 選択した道路のみを残す
    for road in list(roads):
        if road is not selected_road:
            root.remove(road)

    plan_view = selected_road.find("planView")
    if plan_view is None:
        raise ValueError("選択したroadにplanViewが含まれていません")

    if max_geometries is not None:
        geometries = plan_view.findall("geometry")
        if len(geometries) > max_geometries:
            for geom in geometries[max_geometries:]:
                plan_view.remove(geom)

    lanes = selected_road.find("lanes")
    if lanes is not None:
        _prune_lane_sections(lanes, max_lane_sections)

    _simplify_profiles(selected_road)

    header = root.find("header")
    if header is not None:
        _update_length(header, selected_road)

    _cleanup_road_links(selected_road)

    return tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenDRIVEファイルを簡略化して最小区間を抽出します")
    parser.add_argument("input", nargs="?", help="元となる.xodrファイルのパス")
    parser.add_argument("output", nargs="?", help="出力先.xodrファイルのパス")
    parser.add_argument("--road-id", help="保持するroad ID（未指定時は最初のroad）", default=None)
    parser.add_argument(
        "--max-geometries",
        type=int,
        default=1,
        help="planView内で保持するgeometry数",
    )
    parser.add_argument(
        "--max-lane-sections",
        type=int,
        default=1,
        help="lanes内で保持するlaneSection数",
    )
    return parser.parse_args()


def _indent_element(element: ET.Element, level: int = 0) -> None:
    indent_text = "\n" + "  " * level
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = indent_text + "  "
        for child in children:
            _indent_element(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent_text + "  "
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indent_text
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent_text


def main() -> None:
    args = parse_args()
    if args.input is None:
        args.input = input("元の.xodrファイルのパスを入力してください: ").strip()
    if args.output is None:
        args.output = input("出力先.xodrファイルのパスを入力してください: ").strip()

    tree = ET.parse(args.input)
    tree = extract_segment(
        tree,
        args.road_id,
        args.max_geometries,
        args.max_lane_sections,
    )
    _indent_element(tree.getroot())
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"書き込みが完了しました: {args.output}")


if __name__ == "__main__":
    main()
