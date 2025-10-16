import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csv2xodr.line_geometry import build_line_geometry_lookup, _resample_sequence_with_polynomial
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
def test_line_geometry_lookup_fits_curvature_from_geometry():
    meters_to_degrees = 180.0 / (math.pi * 6378137.0)
    radius = 50.0
    rows = []
    for idx, theta in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        arc_length = radius * theta
        x = radius * math.sin(theta)
        y = radius * (1.0 - math.cos(theta))
        rows.append(
            {
                "ライン型地物ID": "arc",
                "Offset[cm]": f"{arc_length * 100.0}",
                "緯度[deg]": f"{y * meters_to_degrees}",
                "経度[deg]": f"{x * meters_to_degrees}",
                "高さ[m]": "0.0",
                "ログ時刻": "t0",
                "Type": "1",
                "Is Retransmission": "false",
            }
        )

    lookup = build_line_geometry_lookup(
        DataFrame(rows),
        offset_mapper=lambda value: value,
        lat0=0.0,
        lon0=0.0,
    )

    geom = lookup["arc"][0]
    assert "curvature" in geom
    expected = 1.0 / radius
    for value in geom["curvature"]:
        assert value == pytest.approx(expected, rel=0.05)


def test_line_geometry_lookup_projects_onto_centerline():
    meters_to_degrees = 180.0 / (math.pi * 6378137.0)
    offsets_cm = [0.0, 250.0, 500.0, 750.0, 1000.0]
    base_x = [0.0, 2.5, 5.0, 7.5, 10.0]
    base_y = [0.0 for _ in base_x]
    width = 3.0
    jitter = [0.4, -0.3, 0.5, -0.2, 0.3]

    rows = []
    for offset, x, y, noise in zip(offsets_cm, base_x, base_y, jitter):
        lat = (y + width + noise) * meters_to_degrees
        lon = x * meters_to_degrees
        rows.append(
            {
                "ライン型地物ID": "LEFT",
                "Offset[cm]": f"{offset}",
                "緯度[deg]": f"{lat}",
                "経度[deg]": f"{lon}",
                "高さ[m]": "0.0",
                "ログ時刻": "proj",
                "Type": "1",
                "Is Retransmission": "false",
            }
        )

    centerline = DataFrame({"s": base_x, "x": base_x, "y": base_y})

    lookup = build_line_geometry_lookup(
        DataFrame(rows),
        offset_mapper=lambda value: value,
        lat0=0.0,
        lon0=0.0,
        centerline=centerline,
    )

    geom = lookup["LEFT"][0]
    s_vals = geom["s"]
    assert s_vals[0] == pytest.approx(0.0)
    assert s_vals[-1] == pytest.approx(10.0)
    assert len(s_vals) >= len(offsets_cm)

    # The reconstructed geometry should follow the centreline heading without jitter.
    for s_val, x_val in zip(s_vals, geom["x"]):
        assert x_val == pytest.approx(s_val, abs=1e-3)

    y_vals = geom["y"]
    avg_y = sum(y_vals) / len(y_vals)
    assert avg_y == pytest.approx(width, abs=0.15)

    diffs = [abs(y_vals[i + 1] - y_vals[i]) for i in range(len(y_vals) - 1)]
    raw_y = [width + noise for noise in jitter]
    raw_diffs = [abs(raw_y[i + 1] - raw_y[i]) for i in range(len(raw_y) - 1)]
    assert sum(diffs) / len(diffs) < sum(raw_diffs) / len(raw_diffs)


def test_resample_sequence_adds_points_for_tight_curves():
    radius = 30.0
    points = []
    for idx in range(6):
        angle = (math.pi / 4.0) * (idx / 5.0)
        s_val = radius * angle
        x_val = radius * math.cos(angle)
        y_val = radius * math.sin(angle)
        points.append((s_val, x_val, y_val, 0.0, None, None))

    resampled = _resample_sequence_with_polynomial(points)
    assert resampled is not None
    s_vals = resampled["s"]
    assert len(s_vals) > len(points)
    gaps = [s_vals[i + 1] - s_vals[i] for i in range(len(s_vals) - 1)]
    assert max(gaps) < 3.0
