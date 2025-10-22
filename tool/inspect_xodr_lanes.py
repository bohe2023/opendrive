#!/usr/bin/env python3
"""Inspect lane composition of OpenDRIVE files.

This helper surfaces the lane counts, types, and width definitions for every
``laneSection`` so that interoperability issues (e.g. MATLAB imports) can be
investigated without manually browsing the XML tree.
"""

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
        width_desc = "no <width>"
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

        balance = "balanced" if left_driving == right_driving else "UNBALANCED"
        print(
            "  driving lanes (L/R): %d / %d -> %s"
            % (left_driving, right_driving, balance)
        )

        if left_counts:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(left_counts.items()))
            print(f"    left types : {summary}")
        else:
            print("    left types : (none)")

        if right_counts:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(right_counts.items()))
            print(f"    right types: {summary}")
        else:
            print("    right types: (none)")

        if center_lanes:
            for lane in center_lanes:
                print("    center    :", _describe_lane(lane))
        else:
            print("    center    : (missing)")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarise lane counts/types for OpenDRIVE laneSections",
    )
    parser.add_argument("paths", nargs="+", type=Path, help=".xodr files to inspect")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for path in args.paths:
        analyse(path)


if __name__ == "__main__":  # pragma: no cover - command-line helper
    main()
