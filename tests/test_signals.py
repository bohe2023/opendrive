import math
from typing import List, Tuple

import pytest

from csv2xodr import signals as signals_mod
from csv2xodr.signals import generate_signals
from csv2xodr.simpletable import DataFrame


def test_generate_signals_jpn_handles_digital_signs():
    df = DataFrame(
        {
            "Offset[cm]": ["100", "200", "300"],
            "最高速度値[km/h]": ["50", "0", "0"],
            "補助標識分類": ["123", "4", "65535"],
        }
    )

    result = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
    )

    assert len(result.signals) == 3
    first, second, third = result.signals

    assert first["s"] == 0.0
    assert first["value"] == 50.0
    assert first["dynamic"] == "no"
    assert first["unit"] == "km/h"
    assert first["supplementary"] == "123"

    assert second["s"] == 1.0
    assert second["dynamic"] == "yes"
    assert second["value"] == 40.0
    assert second["supplementary"] == "4"

    assert third["s"] == 2.0
    assert third["dynamic"] == "yes"
    assert third["value"] == 0.0
    assert "supplementary" not in third

    assert len(result.objects) == 3
    pole = result.objects[0]
    assert pole["type"] == "pole"
    assert pole["orientation"] == "+"


def test_generate_signals_jpn_uses_attribute_flag_for_digital_detection():
    df = DataFrame(
        {
            "Offset[cm]": ["100"],
            "最高速度値[km/h]": [""],
            "標識付加属性フラグ": ["128"],
            "標識情報種別": ["0xC5031700"],
        }
    )

    result = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
    )

    assert len(result.signals) == 1
    signal = result.signals[0]
    assert signal["s"] == 0.0
    assert signal["dynamic"] == "yes"
    assert signal["value"] == 0.0


def test_generate_signals_jpn_extracts_embedded_numeric_speeds():
    df = DataFrame(
        {
            "Offset[cm]": ["0"],
            "最高速度値[km/h]": ["約50km/h"],
        }
    )

    result = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
    )

    assert len(result.signals) == 1
    assert result.signals[0]["value"] == 50.0


def test_generate_signals_handles_grouped_numeric_offsets():
    df = DataFrame(
        {
            "Offset[cm]": ["2,500", "3,000"],
            "最高速度": ["40", "35"],
        }
    )

    result = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
    )

    assert len(result.signals) == 2
    assert result.signals[0]["s"] == 0.0
    assert result.signals[0]["value"] == 40.0
    assert result.signals[1]["s"] == 5.0
    assert result.signals[1]["value"] == 35.0


def test_generate_signals_us_uses_speed_limit_and_shape():
    df = DataFrame(
        {
            "Offset[cm]": ["500", "700"],
            "speed_limit": ["45", ""],
            "shape": ["rectangle", "circle"],
            "type": ["78", "66"],
            "is_digital": ["1", "0"],
            "height[m]": ["1.2", ""],
            "width[m]": ["0.9", ""],
            "Sign Face Elevation[deg]": ["0.5", "-1.0"],
        }
    )

    result = generate_signals(
        df,
        country="US",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_US_SIGN.csv",
        log_fn=lambda message: None,
    )

    assert len(result.signals) == 2
    first, second = result.signals

    assert first["s"] == 0.0
    assert first["value"] == 45.0
    assert first["dynamic"] == "yes"
    assert first["shape"] == "rectangle"
    assert first["unit"] == "mph"
    assert math.isclose(first["height"], 1.2)
    assert math.isclose(first["width"], 0.9)
    assert math.isclose(first["pitch"], math.radians(0.5))
    assert math.isclose(first["zOffset"], signals_mod._SIGN_BOARD_Z_OFFSET_M)

    assert second["s"] == 2.0
    assert second["value"] == 5.0
    assert second["dynamic"] == "no"
    assert second["shape"] == "circle"
    assert second["subtype"] == "max"
    assert math.isclose(second["zOffset"], signals_mod._SIGN_BOARD_Z_OFFSET_M)


def test_generate_signals_us_exports_non_speed_signs():
    df = DataFrame(
        {
            "Offset[cm]": ["1000", "1500"],
            "shape": ["0", "8"],
            "type": ["17", "16"],
            "is_digital": ["0", "1"],
            "height[m]": ["2.5", "1.0"],
            "width[m]": ["6.0", "0.8"],
            "Sign Face Elevation[deg]": ["5.0", "-2.0"],
            "標識情報ID": ["A123", "B456"],
        }
    )

    result = generate_signals(
        df,
        country="US",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_US_SIGN.csv",
        log_fn=lambda message: None,
    )

    assert len(result.signals) == 2
    first, second = result.signals

    assert first["type"] == "information"
    assert first["subtype"] == "17"
    assert first["dynamic"] == "no"
    assert math.isclose(first["height"], 2.5)
    assert math.isclose(first["width"], 6.0)
    assert math.isclose(first["pitch"], math.radians(5.0))
    assert first["name"] == "A123"
    assert first["value"] == 0.0
    assert math.isclose(first["zOffset"], signals_mod._SIGN_BOARD_Z_OFFSET_M)

    assert second["type"] == "information"
    assert second["dynamic"] == "yes"
    assert second["shape"] == "8"
    assert second["subtype"] == "16"
    assert second["name"] == "B456"
    assert math.isclose(second["zOffset"], signals_mod._SIGN_BOARD_Z_OFFSET_M)


def test_generate_signals_us_stacks_signs_for_same_support():
    df = DataFrame(
        {
            "Offset[cm]": ["1000", "1000", "1000"],
            "type": ["17", "17", "16"],
            "shape": ["A", "B", "C"],
            "Instance ID": ["0x1", "0x1", "0x1"],
            "標識情報の配列数": ["3", "3", "3"],
            "height[m]": ["1.0", "0.8", "0.6"],
        }
    )

    result = generate_signals(
        df,
        country="US",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_US_SIGN.csv",
        log_fn=lambda message: None,
    )

    assert len(result.signals) == 3
    first, second, third = result.signals
    base = signals_mod._SIGN_BOARD_Z_OFFSET_M
    gap = signals_mod._SIGN_STACK_GAP_M

    assert math.isclose(first["zOffset"], base)
    assert math.isclose(second["zOffset"], base + 1.0 + gap)
    assert math.isclose(third["zOffset"], base + 1.0 + gap + 0.8 + gap)


def test_generate_signals_projects_latlon_to_centerline():
    lat0 = 35.0
    lon0 = 139.0

    centerline = DataFrame(
        {
            "s": [0.0, 50.0],
            "x": [0.0, 50.0],
            "y": [0.0, 0.0],
        }
    )

    def _xy_to_latlon(x_m: float, y_m: float) -> Tuple[float, float]:
        r = 6378137.0
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)
        lat_rad = lat0_rad + y_m / r
        lon_rad = lon0_rad + x_m / (r * math.cos((lat_rad + lat0_rad) / 2.0))
        return math.degrees(lat_rad), math.degrees(lon_rad)

    latitudes: List[float] = []
    longitudes: List[float] = []
    for x_pos in (10.0, 30.0):
        lat, lon = _xy_to_latlon(x_pos, 0.0)
        latitudes.append(lat)
        longitudes.append(lon)

    df = DataFrame(
        {
            "Offset[cm]": ["1000", "1000"],
            "最高速度値[km/h]": ["50", "60"],
            "緯度": latitudes,
            "経度": longitudes,
        }
    )

    result = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
        centerline=centerline,
        geo_origin=(lat0, lon0),
    )

    assert len(result.signals) == 2
    first, second = result.signals

    assert math.isclose(first["s"], 10.0, abs_tol=1e-3)
    assert math.isclose(second["s"], 30.0, abs_tol=1e-3)


def test_generate_signals_prefers_coordinate2_latlon_columns():
    lat0 = 35.0
    lon0 = 139.0

    centerline = DataFrame(
        {
            "s": [0.0, 50.0],
            "x": [0.0, 50.0],
            "y": [0.0, 0.0],
        }
    )

    def _xy_to_latlon(x_m: float, y_m: float) -> Tuple[float, float]:
        r = 6378137.0
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)
        lat_rad = lat0_rad + y_m / r
        lon_rad = lon0_rad + x_m / (r * math.cos((lat_rad + lat0_rad) / 2.0))
        return math.degrees(lat_rad), math.degrees(lon_rad)

    lat1, lon1 = _xy_to_latlon(0.0, 0.0)
    latitudes2: List[float] = []
    longitudes2: List[float] = []
    for x_pos in (10.0, 30.0):
        lat, lon = _xy_to_latlon(x_pos, 0.0)
        latitudes2.append(lat)
        longitudes2.append(lon)

    df = DataFrame(
        {
            "Offset[cm]": ["1000", "1000"],
            "座標1_緯度[deg]": [lat1, lat1],
            "座標1_経度[deg]": [lon1, lon1],
            "座標2_緯度[deg]": latitudes2,
            "座標2_経度[deg]": longitudes2,
            "最高速度値[km/h]": ["50", "60"],
        }
    )

    result = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
        centerline=centerline,
        geo_origin=(lat0, lon0),
    )

    assert len(result.signals) == 2
    first, second = result.signals

    assert math.isclose(first["s"], 10.0, abs_tol=1e-3)
    assert math.isclose(second["s"], 30.0, abs_tol=1e-3)


def test_generate_signals_uses_non_empty_coordinate_pairs():
    lat0 = 35.0
    lon0 = 139.0

    centerline = DataFrame(
        {
            "s": [0.0, 50.0],
            "x": [0.0, 50.0],
            "y": [0.0, 0.0],
        }
    )

    def _xy_to_latlon(x_m: float, y_m: float) -> Tuple[float, float]:
        r = 6378137.0
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)
        lat_rad = lat0_rad + y_m / r
        lon_rad = lon0_rad + x_m / (r * math.cos((lat_rad + lat0_rad) / 2.0))
        return math.degrees(lat_rad), math.degrees(lon_rad)

    coords = [_xy_to_latlon(pos, 0.0) for pos in (0.0, 10.0, 20.0)]

    df = DataFrame(
        {
            "Offset[cm]": ["1000", "1000", "1000"],
            "最高速度値[km/h]": ["50", "60", "70"],
            "座標1_緯度[deg]": [coords[0][0], "", ""],
            "座標1_経度[deg]": [coords[0][1], "", ""],
            "座標2_緯度[deg]": ["", coords[1][0], ""],
            "座標2_経度[deg]": ["", coords[1][1], ""],
            "座標3_緯度[deg]": ["", "", coords[2][0]],
            "座標3_経度[deg]": ["", "", coords[2][1]],
        }
    )

    result = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
        centerline=centerline,
        geo_origin=(lat0, lon0),
    )

    assert len(result.signals) == 3
    s_values = [signal["s"] for signal in result.signals]
    assert s_values == pytest.approx([0.0, 10.0, 20.0], abs=1e-3)


def test_generate_signals_preserves_duplicate_s_positions():
    df = DataFrame(
        {
            "Offset[cm]": ["1000", "2000", "3000"],
            "最高速度値[km/h]": ["30", "40", "50"],
        }
    )

    # オフセットマッパーが常に同じ弧長を返すよう強制し、座標が異なっても重複
    # オフセットを含む実データを模擬する。
    result = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: 0.0,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
    )

    s_values = [signal["s"] for signal in result.signals]
    assert len(result.signals) == 3
    assert s_values == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
