import math

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

    assert second["s"] == 2.0
    assert second["value"] == 5.0
    assert second["dynamic"] == "no"
    assert second["shape"] == "circle"
    assert second["subtype"] == "max"


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

    assert second["type"] == "information"
    assert second["dynamic"] == "yes"
    assert second["shape"] == "8"
    assert second["subtype"] == "16"
    assert second["name"] == "B456"
