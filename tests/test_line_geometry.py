from pathlib import Path

import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csv2xodr.line_geometry import build_line_geometry_lookup
from csv2xodr.simpletable import DataFrame


def test_build_line_geometry_lookup_deduplicates_retransmissions():
    df = DataFrame(
        [
            {
                "ライン型地物ID": "1.0e2",
                "Offset[cm]": "100",
                "緯度[deg]": "35.0",
                "経度[deg]": "139.0",
                "高さ[m]": "0.0",
                "ログ時刻": "t1",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "1.0e2",
                "Offset[cm]": "200",
                "緯度[deg]": "35.00001",
                "経度[deg]": "139.00001",
                "高さ[m]": "0.0",
                "ログ時刻": "t1",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "1.0e2",
                "Offset[cm]": "300",
                "緯度[deg]": "35.00002",
                "経度[deg]": "139.00002",
                "高さ[m]": "0.0",
                "ログ時刻": "t2",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "1.0e2",
                "Offset[cm]": "400",
                "緯度[deg]": "35.00003",
                "経度[deg]": "139.00003",
                "高さ[m]": "0.0",
                "ログ時刻": "t2",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "1.0e2",
                "Offset[cm]": "100",
                "緯度[deg]": "35.0",
                "経度[deg]": "139.0",
                "高さ[m]": "0.0",
                "ログ時刻": "t3",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "1.0e2",
                "Offset[cm]": "200",
                "緯度[deg]": "35.00001",
                "経度[deg]": "139.00001",
                "高さ[m]": "0.0",
                "ログ時刻": "t3",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "2.0e2",
                "Offset[cm]": "100",
                "緯度[deg]": "35.1",
                "経度[deg]": "139.1",
                "高さ[m]": "0.0",
                "ログ時刻": "t4",
                "Type": "1",
                "Is Retransmission": "true",
            },
        ]
    )

    lookup = build_line_geometry_lookup(
        df, offset_mapper=lambda value: value, lat0=35.0, lon0=139.0
    )

    assert sorted(lookup.keys()) == ["100"], "retransmitted geometry should be skipped"
    assert len(lookup["100"]) == 2

    first, second = lookup["100"]
    assert first["s"][0] == pytest.approx(0.0)
    assert first["s"][-1] == pytest.approx(1.0)
    assert second["s"][0] == pytest.approx(2.0)
    assert second["s"][-1] == pytest.approx(3.0)

    assert first["z"] == [0.0, 0.0]
    assert second["z"] == [0.0, 0.0]


def test_build_line_geometry_lookup_keeps_mixed_retransmissions():
    df = DataFrame(
        [
            {
                "ライン型地物ID": "10",
                "Offset[cm]": "0",
                "緯度[deg]": "35.0",
                "経度[deg]": "139.0",
                "高さ[m]": "0.0",
                "ログ時刻": "mix",
                "Type": "1",
                "Is Retransmission": "true",
            },
            {
                "ライン型地物ID": "10",
                "Offset[cm]": "100",
                "緯度[deg]": "35.00001",
                "経度[deg]": "139.00001",
                "高さ[m]": "0.0",
                "ログ時刻": "mix",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "20",
                "Offset[cm]": "0",
                "緯度[deg]": "35.1",
                "経度[deg]": "139.1",
                "高さ[m]": "0.0",
                "ログ時刻": "only_true",
                "Type": "1",
                "Is Retransmission": "true",
            },
            {
                "ライン型地物ID": "20",
                "Offset[cm]": "100",
                "緯度[deg]": "35.10001",
                "経度[deg]": "139.10001",
                "高さ[m]": "0.0",
                "ログ時刻": "only_true",
                "Type": "1",
                "Is Retransmission": "true",
            },
        ]
    )

    lookup = build_line_geometry_lookup(
        df, offset_mapper=lambda value: value, lat0=35.0, lon0=139.0
    )

    assert "10" in lookup, "mixed retransmission flags should retain the group"
    assert "20" not in lookup, "pure retransmission groups should still be skipped"
    assert lookup["10"][0]["s"][0] == pytest.approx(0.0)


def test_build_line_geometry_lookup_filters_height_outliers():
    df = DataFrame(
        [
            {
                "ライン型地物ID": "50",
                "Offset[cm]": "0",
                "緯度[deg]": "35.0",
                "経度[deg]": "139.0",
                "高さ[m]": "55.0",
                "ログ時刻": "base",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "50",
                "Offset[cm]": "100",
                "緯度[deg]": "35.00001",
                "経度[deg]": "139.00001",
                "高さ[m]": "83886.07",
                "ログ時刻": "base",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "50",
                "Offset[cm]": "200",
                "緯度[deg]": "35.00002",
                "経度[deg]": "139.00002",
                "高さ[m]": "55.5",
                "ログ時刻": "base",
                "Type": "1",
                "Is Retransmission": "false",
            },
        ]
    )

    lookup = build_line_geometry_lookup(
        df, offset_mapper=lambda value: value, lat0=35.0, lon0=139.0
    )

    assert "50" in lookup
    geom = lookup["50"][0]
    assert geom["s"] == pytest.approx([0.0, 1.0, 2.0])
    assert geom["z"] == pytest.approx(
        [55.0, 55.0, 55.5]
    ), "outlier heights should be clamped while keeping continuity"


def test_build_line_geometry_lookup_splits_when_offset_resets():
    df = DataFrame(
        [
            {
                "ライン型地物ID": "500",
                "Offset[cm]": "0",
                "緯度[deg]": "35.0",
                "経度[deg]": "139.0",
                "高さ[m]": "0.0",
                "ログ時刻": "a",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "500",
                "Offset[cm]": "100",
                "緯度[deg]": "35.00001",
                "経度[deg]": "139.00001",
                "高さ[m]": "0.0",
                "ログ時刻": "a",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "500",
                "Offset[cm]": "0",
                "緯度[deg]": "35.1",
                "経度[deg]": "139.1",
                "高さ[m]": "0.0",
                "ログ時刻": "a",
                "Type": "1",
                "Is Retransmission": "false",
            },
            {
                "ライン型地物ID": "500",
                "Offset[cm]": "100",
                "緯度[deg]": "35.10001",
                "経度[deg]": "139.10001",
                "高さ[m]": "0.0",
                "ログ時刻": "a",
                "Type": "1",
                "Is Retransmission": "false",
            },
        ]
    )

    lookup = build_line_geometry_lookup(
        df, offset_mapper=lambda value: value, lat0=35.0, lon0=139.0
    )

    assert "500" in lookup
    geoms = lookup["500"]
    assert len(geoms) == 2, "offset resets should start a new polyline"

    first, second = geoms
    assert first["s"] == pytest.approx([0.0, 1.0])
    assert second["s"] == pytest.approx([0.0, 1.0])
    assert first["y"][0] != pytest.approx(second["y"][0])
