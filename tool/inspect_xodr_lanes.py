#!/usr/bin/env python3
"""OpenDRIVEファイルのレーン構成を調査するユーティリティ。

各 ``laneSection`` のレーン数・種類・幅定義を一覧化し、XMLを手作業で追わずに
MATLAB などとの相互運用上の問題を洗い出せるようにする。"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


def _collect_lanes(section: ET.Element, side: str) -> List[ET.Element]:
    block = section.find(side)
    if block is None:
        return []
    return list(block.findall("lane"))


def _classify_lanes(lanes: Iterable[ET.Element]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for lane in lanes:
        counts[lane.get("type", "").lower()] += 1
    return counts


def _describe_lane(lane: ET.Element) -> str:
    lane_id = lane.get("id", "?")
    lane_type = lane.get("type", "?")
    widths = lane.findall("width")
    width_desc: str
    if not widths:
        width_desc = "<width> 未定義"
    else:
        coeffs = widths[0].attrib
        width_desc = "a={a} b={b} c={c} d={d}".format(
            a=coeffs.get("a", "?"),
            b=coeffs.get("b", "?"),
            c=coeffs.get("c", "?"),
            d=coeffs.get("d", "?"),
        )
    return f"id={lane_id} type={lane_type} ({width_desc})"


def _iter_sections(tree: ET.ElementTree) -> Iterable[Tuple[int, ET.Element]]:
    root = tree.getroot()
    for idx, section in enumerate(root.findall(".//laneSection")):
        yield idx, section


def analyse(path: Path) -> None:
    tree = ET.parse(path)

    print(f"File: {path}")
    for index, section in _iter_sections(tree):
        start_s = section.get("s", "?")
        print(f"\n[laneSection #{index} @ s={start_s}]")

        left_lanes = _collect_lanes(section, "left")
        right_lanes = _collect_lanes(section, "right")
        center_lanes = _collect_lanes(section, "center")

        left_counts = _classify_lanes(left_lanes)
        right_counts = _classify_lanes(right_lanes)

        left_driving = left_counts.get("driving", 0)
        right_driving = right_counts.get("driving", 0)

        balance = "均衡" if left_driving == right_driving else "不均衡"
        print(
            "  走行レーン数 (左/右): %d / %d -> %s"
            % (left_driving, right_driving, balance)
        )

        if left_counts:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(left_counts.items()))
            print(f"    左側タイプ : {summary}")
        else:
            print("    左側タイプ : (なし)")

        if right_counts:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(right_counts.items()))
            print(f"    右側タイプ : {summary}")
        else:
            print("    右側タイプ : (なし)")

        if center_lanes:
            for lane in center_lanes:
                print("    中央      :", _describe_lane(lane))
        else:
            print("    中央      : (未設定)")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="OpenDRIVEのlaneSectionごとにレーン数と種類を集計する",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="調査対象となる.xodrファイル")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for path in args.paths:
        analyse(path)


if __name__ == "__main__":  # pragma: no cover - コマンドライン補助
    main()
