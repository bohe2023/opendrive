from csv2xodr.normalize.core import _merge_geometry_segments


def test_merge_geometry_segments_skips_zero_length_entries():
    segments = [
        {
            "s": 0.0,
            "x": 0.0,
            "y": 0.0,
            "hdg": 0.0,
            "length": 5.0,
            "curvature": 0.0,
        },
        {
            "s": 5.0,
            "x": 5.0,
            "y": 0.0,
            "hdg": 0.0,
            "length": 0.0,
            "curvature": 0.0,
        },
        {
            "s": 5.0,
            "x": 5.0,
            "y": 0.0,
            "hdg": 0.5,
            "length": 2.0,
            "curvature": 0.1,
        },
    ]

    merged = _merge_geometry_segments(segments)

    assert len(merged) == 2
    assert all(seg["length"] > 0 for seg in merged)
    assert merged[0]["length"] == 5.0
    assert merged[1]["s"] == 5.0
    assert merged[1]["length"] == 2.0
