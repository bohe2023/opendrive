import pytest

from csv2xodr.normalize.core import build_offset_mapper
from csv2xodr.simpletable import DataFrame


def test_build_offset_mapper_extrapolates_beyond_centerline():
    centerline = DataFrame(
        {
            "s": [0.0, 100.0, 200.0],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 0.0, 0.0],
            "s_offset": [0.0, 100.0, 200.0],
        }
    )

    mapper = build_offset_mapper(centerline)

    assert mapper(50.0) == pytest.approx(50.0)
    assert mapper(-50.0) == pytest.approx(-50.0)
    assert mapper(250.0) == pytest.approx(250.0)
    assert mapper(300.0) == pytest.approx(300.0)
