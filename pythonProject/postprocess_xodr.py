"""OpenDRIVE生成物の中心レーン定義を調整する後処理ユーティリティ。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional


def _iter_center_lanes(tree: ET.ElementTree) -> Iterable[ET.Element]:
    """center要素配下のlaneノードを順に返す。"""

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
) -> bool:
    """中心レーンの ``type`` と ``<width>`` を正規化する。"""

    tree = ET.parse(path)
    updated = False

    for lane in _iter_center_lanes(tree):
        lane_updated = False

        if force_lane_type is not None and lane.get("type") != force_lane_type:
            lane.set("type", force_lane_type)
            lane_updated = True

        # 参照線として扱うため幅ノードはすべて削除する。
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
    """単一XODRファイルを後処理し変更有無を返す。"""

    try:
        changed = ensure_center_lane_width(path, force_lane_type="none")
    except FileNotFoundError:
        if verbose:
            print(f"[SKIP] XODRファイルが見つかりません: {path}")
        return False

    if changed and verbose:
        print(f"[OK] 中央レーン定義を更新しました: {path}")
    elif verbose:
        print(f"[OK] 中央レーン定義は既に条件を満たしています: {path}")
    return changed


def patch_paths(paths: Iterable[Path], *, verbose: bool = True) -> int:
    """複数パスを処理し更新数を返す。"""

    touched = 0
    for path in paths:
        if patch_file(path, verbose=verbose):
            touched += 1
    return touched


def main(argv: Optional[Iterable[str]] = None) -> None:
    """必要に応じて手動実行できる簡易CLI。"""

    import argparse

    parser = argparse.ArgumentParser(description="中央レーンのタイプと幅の定義を正規化します")
    parser.add_argument("paths", nargs="+", help="修正対象のXODRファイルパス")
    parser.add_argument(
        "--quiet", action="store_true", help="詳細を表示せずに終了コードのみ返します"
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    path_objs = [Path(p) for p in args.paths]

    modified = patch_paths(path_objs, verbose=not args.quiet)
    raise SystemExit(0 if modified >= 0 else 1)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    main()

