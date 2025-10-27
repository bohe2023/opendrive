#!/usr/bin/env python3
"""デバッグ用途に最小構成のOpenDRIVEサンプルを生成する補助スクリプト。

本来の変換パイプラインと同じ構造を保ちつつジオメトリを単純化し、
シナリオごとに挙動の異なる組み合わせを用意する。生成した ``.xodr`` を
MATLAB などへ個別に読み込むことで相互運用性の問題を切り分けやすくする。"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csv2xodr.lane_spec import build_lane_spec
from csv2xodr.simpletable import DataFrame
from csv2xodr.writer.xodr_writer import write_xodr
from pythonProject.postprocess_xodr import patch_file


@dataclass
class Scenario:
    """シナリオを書き出すために必要な最小限のデータをまとめたコンテナ。"""

    name: str
    centerline: DataFrame
    sections: List[Dict[str, float]]
    lane_topology: Dict[str, object]
    geometry_segments: Optional[List[Dict[str, float]]] = None


_DEF_LINE_POSITIONS: Dict[str, float] = {}


def _lane_segment(start: float, end: float, width: float) -> Dict[str, object]:
    return {
        "start": float(start),
        "end": float(end),
        "width": float(width),
        "successors": [],
        "predecessors": [],
        "line_positions": dict(_DEF_LINE_POSITIONS),
    }


def _make_centerline(points: Iterable[Iterable[float]]) -> DataFrame:
    """``s/x/y/hdg`` 列のみを持つ最小限の :class:`DataFrame` を構築する。"""

    s_vals: List[float] = []
    x_vals: List[float] = []
    y_vals: List[float] = []
    hdg_vals: List[float] = []

    for entry in points:
        s, x, y, hdg = entry
        s_vals.append(float(s))
        x_vals.append(float(x))
        y_vals.append(float(y))
        hdg_vals.append(float(hdg))

    return DataFrame({"s": s_vals, "x": x_vals, "y": y_vals, "hdg": hdg_vals})


def _make_lane_topology(
    left_lanes: List[List[Dict[str, object]]],
    right_lanes: List[List[Dict[str, object]]],
) -> Dict[str, object]:
    """:func:`build_lane_spec` と互換性のあるレーン構造データを生成する。"""

    groups: Dict[str, List[str]] = {}
    lanes: Dict[str, Dict[str, object]] = {}
    lane_count = 0

    for idx, segments in enumerate(left_lanes, start=1):
        base_id = f"L{idx}"
        uid = f"{base_id}:{idx}"
        groups[base_id] = [uid]
        lanes[uid] = {
            "base_id": base_id,
            "lane_no": idx,
            "segments": list(segments),
        }
        lane_count += 1

    for idx, segments in enumerate(right_lanes, start=1):
        base_id = f"R{idx}"
        uid = f"{base_id}:{idx}"
        groups[base_id] = [uid]
        lanes[uid] = {
            "base_id": base_id,
            "lane_no": -idx,
            "segments": list(segments),
        }
        lane_count += 1

    return {"lane_count": lane_count, "groups": groups, "lanes": lanes}


def _integrate_spiral(
    length: float,
    curvature_start: float,
    curvature_end: float,
    x0: float,
    y0: float,
    hdg0: float,
) -> Dict[str, float]:
    """平面クロソイドを数値積分して終端姿勢を返す。"""

    if length <= 0.0:
        return {"x": float(x0), "y": float(y0), "hdg": float(hdg0)}

    # 短いスパイラルにも十分なサンプリングを与えつつ、長距離では過密になりすぎない
    # よう積分ステップ数を制限する。
    steps = max(2000, int(length * 200))
    ds_nominal = length / steps
    curvature_delta = curvature_end - curvature_start
    curvature_slope = curvature_delta / length

    x = float(x0)
    y = float(y0)
    hdg = float(hdg0)
    travelled = 0.0

    while travelled < length:
        remaining = length - travelled
        ds = ds_nominal if remaining > ds_nominal else remaining
        curvature = curvature_start + curvature_slope * travelled
        x += math.cos(hdg) * ds
        y += math.sin(hdg) * ds
        hdg += curvature * ds
        travelled += ds

    hdg = hdg0 + curvature_start * length + 0.5 * curvature_delta * length
    return {"x": x, "y": y, "hdg": hdg}


def _integrate_arc(
    length: float,
    curvature: float,
    x0: float,
    y0: float,
    hdg0: float,
) -> Dict[str, float]:
    """一定曲率の円弧を積分して終端姿勢を返す。"""

    if length <= 0.0:
        return {"x": float(x0), "y": float(y0), "hdg": float(hdg0)}

    if abs(curvature) <= 1e-9:
        x = float(x0) + math.cos(hdg0) * length
        y = float(y0) + math.sin(hdg0) * length
        return {"x": x, "y": y, "hdg": float(hdg0)}

    delta = curvature * length
    x = x0 + (math.sin(hdg0 + delta) - math.sin(hdg0)) / curvature
    y = y0 - (math.cos(hdg0 + delta) - math.cos(hdg0)) / curvature
    hdg = hdg0 + delta
    return {"x": x, "y": y, "hdg": hdg}


def _extreme_uturn_parameters() -> Dict[str, float]:
    """極端なUターン系シナリオで共有する基本パラメータを返す。"""

    first_line = 120.0
    spiral_length = 8.0
    last_line = 120.0
    curvature_start = 0.0
    curvature_end = 2.0 * math.pi / spiral_length  # およそ180度の方位変化になる
    return {
        "first_line": first_line,
        "turn_length": spiral_length,
        "last_line": last_line,
        "curvature_start": curvature_start,
        "curvature_end": curvature_end,
    }


def _straight_single_section() -> Scenario:
    length = 30.0
    centerline = _make_centerline([(0.0, 0.0, 0.0, 0.0), (length, length, 0.0, 0.0)])
    sections = [{"s0": 0.0, "s1": length}]
    lane_topology = _make_lane_topology(
        left_lanes=[[_lane_segment(0.0, length, 3.5)]],
        right_lanes=[[_lane_segment(0.0, length, 3.5)]],
    )
    return Scenario("straight_single_section", centerline, sections, lane_topology)


def _straight_multi_section() -> Scenario:
    length = 60.0
    centerline = _make_centerline([(0.0, 0.0, 0.0, 0.0), (length, length, 0.0, 0.0)])
    sections = [
        {"s0": 0.0, "s1": 20.0},
        {"s0": 20.0, "s1": 40.0},
        {"s0": 40.0, "s1": length},
    ]
    left_lanes = [
        [
            _lane_segment(0.0, 20.0, 3.5),
            _lane_segment(20.0, 40.0, 3.3),
            _lane_segment(40.0, length, 3.6),
        ],
        [_lane_segment(0.0, length, 3.4)],
    ]
    right_lanes = [
        [
            _lane_segment(0.0, 30.0, 3.5),
            _lane_segment(30.0, length, 3.2),
        ],
        [_lane_segment(0.0, length, 3.4)],
    ]
    lane_topology = _make_lane_topology(left_lanes, right_lanes)
    return Scenario("straight_multi_section", centerline, sections, lane_topology)


def _curved_arc() -> Scenario:
    radius = 40.0
    theta = math.radians(45.0)
    length = radius * theta
    x_end = radius * math.sin(theta)
    y_end = radius * (1.0 - math.cos(theta))
    centerline = _make_centerline([(0.0, 0.0, 0.0, 0.0), (length, x_end, y_end, theta)])
    sections = [{"s0": 0.0, "s1": length}]
    geometry_segments = [
        {
            "s": 0.0,
            "x": 0.0,
            "y": 0.0,
            "hdg": 0.0,
            "length": length,
            "curvature": 1.0 / radius,
            "curvature_start": 1.0 / radius,
            "curvature_end": 1.0 / radius,
        }
    ]
    lane_topology = _make_lane_topology(
        left_lanes=[[_lane_segment(0.0, length, 3.5)]],
        right_lanes=[[_lane_segment(0.0, length, 3.5)]],
    )
    return Scenario("curved_arc", centerline, sections, lane_topology, geometry_segments)


def _extreme_spiral_uturn() -> Scenario:
    """直線-スパイラル-直線構成の急峻なUターンシナリオを構築する。"""

    params = _extreme_uturn_parameters()
    first_line = params["first_line"]
    spiral_length = params["turn_length"]
    last_line = params["last_line"]
    curvature_start = params["curvature_start"]
    curvature_end = params["curvature_end"]

    spiral_end = _integrate_spiral(
        spiral_length,
        curvature_start,
        curvature_end,
        x0=first_line,
        y0=0.0,
        hdg0=0.0,
    )

    total_length = first_line + spiral_length + last_line
    final_x = spiral_end["x"] + last_line * math.cos(spiral_end["hdg"])
    final_y = spiral_end["y"] + last_line * math.sin(spiral_end["hdg"])

    # 中心線テーブルへ中間スパイラルを幾つか追加し、離散化に依存する後段ツールでも
    # 急激な曲率変化を把握できるようにする。
    spiral_mid_s = first_line + spiral_length / 2.0
    spiral_mid = _integrate_spiral(
        spiral_length / 2.0,
        curvature_start,
        (curvature_start + curvature_end) / 2.0,
        x0=first_line,
        y0=0.0,
        hdg0=0.0,
    )

    centerline = _make_centerline(
        [
            (0.0, 0.0, 0.0, 0.0),
            (first_line, first_line, 0.0, 0.0),
            (spiral_mid_s, spiral_mid["x"], spiral_mid["y"], spiral_mid["hdg"]),
            (first_line + spiral_length, spiral_end["x"], spiral_end["y"], spiral_end["hdg"]),
            (total_length, final_x, final_y, spiral_end["hdg"]),
        ]
    )

    sections = [{"s0": 0.0, "s1": total_length}]
    lane_topology = _make_lane_topology(
        left_lanes=[[_lane_segment(0.0, total_length, 3.5)]],
        right_lanes=[[_lane_segment(0.0, total_length, 3.5)]],
    )

    geometry_segments = [
        {
            "s": 0.0,
            "x": 0.0,
            "y": 0.0,
            "hdg": 0.0,
            "length": first_line,
            "curvature": 0.0,
        },
        {
            "s": first_line,
            "x": first_line,
            "y": 0.0,
            "hdg": 0.0,
            "length": spiral_length,
            "curvature_start": curvature_start,
            "curvature_end": curvature_end,
        },
        {
            "s": first_line + spiral_length,
            "x": spiral_end["x"],
            "y": spiral_end["y"],
            "hdg": spiral_end["hdg"],
            "length": last_line,
            "curvature": 0.0,
        },
    ]

    return Scenario(
        "extreme_spiral_uturn",
        centerline,
        sections,
        lane_topology,
        geometry_segments,
    )


def _extreme_arc_uturn() -> Scenario:
    """極端なUターンを平均曲率の円弧で近似したシナリオを構築する。"""

    params = _extreme_uturn_parameters()
    first_line = params["first_line"]
    arc_length = params["turn_length"]
    last_line = params["last_line"]
    curvature_avg = 0.5 * (params["curvature_start"] + params["curvature_end"])

    arc_mid = _integrate_arc(
        arc_length / 2.0,
        curvature_avg,
        x0=first_line,
        y0=0.0,
        hdg0=0.0,
    )
    arc_end = _integrate_arc(
        arc_length,
        curvature_avg,
        x0=first_line,
        y0=0.0,
        hdg0=0.0,
    )

    total_length = first_line + arc_length + last_line
    final_x = arc_end["x"] + last_line * math.cos(arc_end["hdg"])
    final_y = arc_end["y"] + last_line * math.sin(arc_end["hdg"])

    centerline = _make_centerline(
        [
            (0.0, 0.0, 0.0, 0.0),
            (first_line, first_line, 0.0, 0.0),
            (first_line + arc_length / 2.0, arc_mid["x"], arc_mid["y"], arc_mid["hdg"]),
            (first_line + arc_length, arc_end["x"], arc_end["y"], arc_end["hdg"]),
            (total_length, final_x, final_y, arc_end["hdg"]),
        ]
    )

    sections = [{"s0": 0.0, "s1": total_length}]
    lane_topology = _make_lane_topology(
        left_lanes=[[_lane_segment(0.0, total_length, 3.5)]],
        right_lanes=[[_lane_segment(0.0, total_length, 3.5)]],
    )

    geometry_segments = [
        {
            "s": 0.0,
            "x": 0.0,
            "y": 0.0,
            "hdg": 0.0,
            "length": first_line,
            "curvature": 0.0,
        },
        {
            "s": first_line,
            "x": first_line,
            "y": 0.0,
            "hdg": 0.0,
            "length": arc_length,
            "curvature": curvature_avg,
        },
        {
            "s": first_line + arc_length,
            "x": arc_end["x"],
            "y": arc_end["y"],
            "hdg": arc_end["hdg"],
            "length": last_line,
            "curvature": 0.0,
        },
    ]

    return Scenario(
        "extreme_arc_uturn",
        centerline,
        sections,
        lane_topology,
        geometry_segments,
    )


def _iter_scenarios() -> Iterable[Scenario]:
    yield _straight_single_section()
    yield _straight_multi_section()
    yield _curved_arc()
    yield _extreme_spiral_uturn()
    yield _extreme_arc_uturn()


def generate_samples(output_dir: Path, *, patch: bool = True) -> List[Path]:
    """全シナリオを *output_dir* に書き出し生成されたファイル一覧を返す。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for scenario in _iter_scenarios():
        lane_specs = build_lane_spec(
            scenario.sections,
            scenario.lane_topology,
            defaults={},
            lane_div_df=None,
        )
        out_path = output_dir / f"{scenario.name}.xodr"
        write_xodr(
            scenario.centerline,
            scenario.sections,
            lane_specs,
            out_path,
            geometry_segments=scenario.geometry_segments,
        )
        if patch:
            patch_file(out_path, verbose=False)
        written.append(out_path)

    return written


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="最小構成のOpenDRIVEサンプルを生成する")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/minimal_scenarios"),
        help="生成した.xodrファイルを配置するディレクトリ",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="中央レーン幅を付加する後処理をスキップする",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    created = generate_samples(args.output, patch=not args.no_patch)
    for path in created:
        print(path)


if __name__ == "__main__":  # pragma: no cover - スクリプトの入口
    main()
