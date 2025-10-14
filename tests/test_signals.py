from csv2xodr.signals import generate_signals
from csv2xodr.simpletable import DataFrame


def test_generate_signals_jpn_handles_digital_signs():
    df = DataFrame(
        {
            "Offset[cm]": ["100", "200"],
            "最高速度値[km/h]": ["50", "0"],
            "補助標識分類": ["123", "65535"],
        }
    )

    signals = generate_signals(
        df,
        country="JPN",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_ZGM_SIGN_INFO.csv",
        log_fn=lambda message: None,
    )

    assert len(signals) == 2
    first, second = signals

    assert first["s"] == 0.0
    assert first["value"] == 50.0
    assert first["dynamic"] == "no"
    assert first["unit"] == "km/h"
    assert first["supplementary"] == "123"

    assert second["s"] == 1.0
    assert second["dynamic"] == "yes"
    assert second["value"] == 0.0
    assert "supplementary" not in second


def test_generate_signals_us_uses_speed_limit_and_shape():
    df = DataFrame(
        {
            "Offset[cm]": ["500", "700"],
            "speed_limit": ["45", ""],
            "shape": ["rectangle", "circle"],
            "type": ["78", "66"],
            "is_digital": ["1", "0"],
        }
    )

    signals = generate_signals(
        df,
        country="US",
        offset_mapper=lambda value: value,
        sign_filename="PROFILETYPE_MPU_US_SIGN.csv",
        log_fn=lambda message: None,
    )

    assert len(signals) == 2
    first, second = signals

    assert first["s"] == 0.0
    assert first["value"] == 45.0
    assert first["dynamic"] == "yes"
    assert first["shape"] == "rectangle"
    assert first["unit"] == "mph"

    assert second["s"] == 2.0
    assert second["value"] == 5.0
    assert second["dynamic"] == "no"
    assert second["shape"] == "circle"
    assert second["subtype"] == "max"
