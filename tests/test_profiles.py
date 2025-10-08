import math

from csv2xodr.normalize.core import (
    build_curvature_profile,
    build_elevation_profile_from_slopes,
    build_geometry_segments,
    build_shoulder_profile,
    build_slope_profile,
    build_superelevation_profile,
    _advance_pose,
    _merge_geometry_segments,
    _normalize_angle,
)
from csv2xodr.lane_spec import apply_shoulder_profile
from csv2xodr.simpletable import DataFrame


def test_build_slope_profiles_and_elevation():
    df = DataFrame(
        {
            "Offset[cm]": [0, 100],
            "End Offset[cm]": [100, 200],
            "縦断勾配値[%]": ["1.0", "2.0"],
            "横断勾配値[%]": ["3.0", "4.0"],
            "Is Retransmission": ["False", "False"],
        }
    )

    profiles = build_slope_profile(df)
    longitudinal = profiles["longitudinal"]
    superelev = profiles["superelevation"]

    assert longitudinal == [
        {"s0": 0.0, "s1": 1.0, "grade": 0.01},
        {"s0": 1.0, "s1": 2.0, "grade": 0.02},
    ]
    assert superelev == [
        {"s0": 0.0, "s1": 1.0, "angle": 0.03},
        {"s0": 1.0, "s1": 2.0, "angle": 0.04},
    ]

    elevation = build_elevation_profile_from_slopes(longitudinal)
    assert elevation[0] == {"s": 0.0, "a": 0.0, "b": 0.01, "c": 0.0, "d": 0.0}
    assert elevation[1] == {"s": 1.0, "a": 0.01, "b": 0.02, "c": 0.0, "d": 0.0}

    sup_profile = build_superelevation_profile(superelev)
    assert sup_profile == [
        {"s": 0.0, "a": 0.03, "b": 0.0, "c": 0.0, "d": 0.0},
        {"s": 1.0, "a": 0.04, "b": 0.0, "c": 0.0, "d": 0.0},
    ]


def test_geometry_segments_from_curvature():
    # The centreline follows a straight 1 m segment and then a 0.1 rad/m arc
    # so the curvature profile is geometrically consistent with the anchor
    # points.
    center = DataFrame(
        {
            "s": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 1.9983341664682815],
            "y": [0.0, 0.0, 0.049958347219741794],
            "hdg": [0.0, 0.0, 0.1],
        }
    )

    curvature_df = DataFrame(
        {
            "Offset[cm]": [0, 100],
            "End Offset[cm]": [100, 200],
            "曲率値[rad/m]": [0.0, 0.1],
            "Is Retransmission": ["False", "False"],
        }
    )

    curvature_segments, _ = build_curvature_profile(curvature_df)
    geometry = build_geometry_segments(center, curvature_segments)

    assert len(geometry) == 2
    assert geometry[0]["length"] == 1.0
    assert geometry[0]["curvature"] == 0.0
    assert geometry[1]["curvature"] == 0.1


def test_geometry_segments_respect_threshold():
    center = DataFrame({
        "s": [0.0, 1.0],
        "x": [0.0, 1.0],
        "y": [0.0, 0.0],
        "hdg": [0.0, 0.0],
    })

    curvature_df = DataFrame(
        {
            "Offset[cm]": [0],
            "End Offset[cm]": [100],
            "曲率値[rad/m]": [0.2],
            "Is Retransmission": ["False"],
        }
    )

    curvature_segments, _ = build_curvature_profile(curvature_df)

    strict_geometry = build_geometry_segments(
        center,
        curvature_segments,
        max_endpoint_deviation=0.01,
    )

    assert len(strict_geometry) >= 1

    seg = strict_geometry[0]
    if abs(seg.get("curvature", 0.0)) <= 1e-12:
        end_x = seg["x"] + seg["length"] * math.cos(seg["hdg"])
        end_y = seg["y"] + seg["length"] * math.sin(seg["hdg"])
    else:
        radius = 1.0 / seg["curvature"]
        end_hdg = seg["hdg"] + seg["curvature"] * seg["length"]
        end_x = seg["x"] + radius * (math.sin(end_hdg) - math.sin(seg["hdg"]))
        end_y = seg["y"] - radius * (math.cos(end_hdg) - math.cos(seg["hdg"]))

    assert math.hypot(end_x - 1.0, end_y - 0.0) <= 0.01 + 1e-6

    relaxed_geometry = build_geometry_segments(
        center,
        curvature_segments,
        max_endpoint_deviation=0.5,
    )

    assert len(relaxed_geometry) >= 1


def test_geometry_segments_remain_continuous():
    center = DataFrame({
        "s": [0.0, 5.0, 10.0, 15.0],
        "x": [0.0, 5.0, 10.0, 15.0],
        "y": [0.0, 0.0, 0.0, 0.0],
        "hdg": [0.0, 0.0, 0.0, 0.0],
    })

    curvature_df = DataFrame(
        {
            "Offset[cm]": [0, 500, 1000],
            "End Offset[cm]": [500, 1000, 1500],
            "曲率値[rad/m]": [0.0, 0.002, 0.0],
            "Is Retransmission": ["False", "False", "False"],
        }
    )

    curvature_segments, _ = build_curvature_profile(curvature_df)
    geometry = build_geometry_segments(
        center,
        curvature_segments,
        max_endpoint_deviation=0.5,
    )

    assert len(geometry) >= 3
    assert math.isclose(
        sum(seg["length"] for seg in geometry),
        center["s"].iloc[-1],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert any(abs(seg.get("curvature", 0.0)) > 0 for seg in geometry)

    def _apply(seg):
        length = seg["length"]
        heading = seg["hdg"]
        curv = seg.get("curvature", 0.0)
        if abs(curv) <= 1e-12:
            end_x = seg["x"] + length * math.cos(heading)
            end_y = seg["y"] + length * math.sin(heading)
            end_hdg = heading
        else:
            radius = 1.0 / curv
            end_hdg = heading + curv * length
            end_x = seg["x"] + radius * (math.sin(end_hdg) - math.sin(heading))
            end_y = seg["y"] - radius * (math.cos(end_hdg) - math.cos(heading))
        return end_x, end_y, end_hdg

    for idx in range(len(geometry) - 1):
        end_x, end_y, end_hdg = _apply(geometry[idx])
        next_seg = geometry[idx + 1]
        assert math.isclose(end_x, next_seg["x"], abs_tol=1e-9)
        assert math.isclose(end_y, next_seg["y"], abs_tol=1e-9)
        assert math.isclose(
            _normalize_angle(end_hdg - next_seg["hdg"]),
            0.0,
            abs_tol=1e-9,
        )


def test_merge_geometry_segments_snaps_small_drift():
    segments = [
        {"s": 0.0, "x": 0.0, "y": 0.0, "hdg": 0.0, "length": 5.0, "curvature": 0.001},
        {
            "s": 5.0,
            "x": 5.003,
            "y": 0.002,
            "hdg": 0.0018,
            "length": 4.0,
            "curvature": 0.0012,
        },
    ]

    merged = _merge_geometry_segments(segments)
    assert len(merged) == 2

    expected_x, expected_y, expected_hdg = _advance_pose(
        segments[0]["x"],
        segments[0]["y"],
        segments[0]["hdg"],
        segments[0]["curvature"],
        segments[0]["length"],
    )

    second = merged[1]
    assert math.isclose(second["x"], expected_x, abs_tol=1e-6)
    assert math.isclose(second["y"], expected_y, abs_tol=1e-6)
    assert math.isclose(
        _normalize_angle(second["hdg"] - expected_hdg),
        0.0,
        abs_tol=1e-6,
    )


def test_geometry_segments_are_densified_for_long_spans():
    center = DataFrame(
        {
            "s": [0.0, 50.0, 100.0],
            "x": [0.0, 50.0, 100.0],
            "y": [0.0, 0.0, 0.0],
            "hdg": [0.0, 0.0, 0.0],
        }
    )

    geometry = build_geometry_segments(
        center,
        [{"s0": 0.0, "s1": 100.0, "curvature": 0.0}],
        max_endpoint_deviation=0.5,
    )

    assert len(geometry) >= 50
    assert all(seg["length"] <= 2.0 + 1e-9 for seg in geometry)


def test_curvature_profile_uses_shape_index_segments():
    meters_to_degrees = 180.0 / (math.pi * 6378137.0)
    coord_by_index = {
        0: 0.0,
        1: 1.0,
        2: 3.0,
        3: 6.0,
        4: 10.0,
    }

    curvature_df = DataFrame(
        {
            "Path Id": ["1"] * 7,
            "Lane Number": ["1"] * 7,
            "Offset[cm]": [0, 0, 0, 0, 0, 0, 0],
            "End Offset[cm]": [1000, 1000, 1000, 1000, 1000, 1000, 1000],
            "形状インデックス": [0, 1, 2, 2, 3, 3, 4],
            "曲率値[rad/m]": [0.6, -0.6, 0.6, -0.2, -0.6, -0.6, 0.6],
            "緯度[deg]": [0.0] * 7,
            "経度[deg]": [coord_by_index[idx] * meters_to_degrees for idx in [0, 1, 2, 2, 3, 3, 4]],
            "Is Retransmission": [
                "False",
                "False",
                "False",
                "False",
                "True",
                "False",
                "False",
            ],
        }
    )

    curvature_segments, samples = build_curvature_profile(
        curvature_df,
        offset_mapper=lambda value: value,
        geo_origin=(0.0, 0.0),
    )

    assert len(curvature_segments) == 4
    assert samples
    assert len(samples) >= 5

    spans = [seg["s1"] - seg["s0"] for seg in curvature_segments]
    expected_spans = [1.0, 2.0, 3.0, 4.0]
    assert all(math.isclose(span, exp, abs_tol=1e-9) for span, exp in zip(spans, expected_spans))

    curvatures = [seg["curvature"] for seg in curvature_segments]
    assert all(math.isclose(curv, expected) for curv, expected in zip(curvatures, [0.6, -0.6, 0.2, -0.6]))

    center = DataFrame(
        {
            "s": [0.0, 0.5, 1.0, 1.5, 2.0],
            "x": [0.0, 0.5, 1.0, 1.5, 2.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "hdg": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    geometry = build_geometry_segments(
        center,
        curvature_segments,
        max_endpoint_deviation=0.01,
        max_segment_length=0.5,
    )

    geometry_curvatures = [seg.get("curvature", 0.0) for seg in geometry]
    transitions = sum(
        1
        for i in range(1, len(geometry_curvatures))
        if geometry_curvatures[i - 1] * geometry_curvatures[i] < 0
    )
    assert transitions >= 2


def test_curvature_profile_averages_duplicate_shape_indices():
    curvature_df = DataFrame(
        {
            "Offset[cm]": [0, 0, 0, 0],
            "End Offset[cm]": [200, 200, 200, 200],
            "Lane Number": ["A", "A", "A", "A"],
            "形状インデックス": [0, 0, 1, 2],
            "曲率値[rad/m]": [0.4, 0.8, 1.0, 1.2],
            "Is Retransmission": ["False", "False", "False", "False"],
        }
    )

    curvature_segments, _ = build_curvature_profile(curvature_df)

    assert len(curvature_segments) == 2

    first, second = curvature_segments
    assert math.isclose(first["curvature"], 0.6)
    assert math.isclose(second["curvature"], 1.0)


def test_geometry_segments_honours_custom_densify_threshold():
    center = DataFrame(
        {
            "s": [0.0, 50.0, 100.0],
            "x": [0.0, 50.0, 100.0],
            "y": [0.0, 0.0, 0.0],
            "hdg": [0.0, 0.0, 0.0],
        }
    )

    geometry = build_geometry_segments(
        center,
        [{"s0": 0.0, "s1": 100.0, "curvature": 0.0}],
        max_endpoint_deviation=0.5,
        max_segment_length=5.0,
    )

    assert len(geometry) >= 20
    assert all(seg["length"] <= 5.0 + 1e-9 for seg in geometry)

def test_apply_shoulder_profile_adds_lanes():
    lane_sections = [
        {"s0": 0.0, "s1": 10.0, "left": [{"id": 1, "width": 3.5, "roadMark": {}, "predecessors": [], "successors": [], "type": "driving"}], "right": [{"id": -1, "width": 3.5, "roadMark": {}, "predecessors": [], "successors": [], "type": "driving"}]},
        {"s0": 10.0, "s1": 20.0, "left": [{"id": 1, "width": 3.5, "roadMark": {}, "predecessors": [], "successors": [], "type": "driving"}], "right": [{"id": -1, "width": 3.5, "roadMark": {}, "predecessors": [], "successors": [], "type": "driving"}]},
    ]

    shoulder_df = DataFrame(
        {
            "Offset[cm]": [0, 1000],
            "End Offset[cm]": [1000, 2000],
            "左側路肩幅員値[cm]": [200, 250],
            "右側路肩幅員値[cm]": [150, 100],
            "Is Retransmission": ["False", "False"],
        }
    )

    shoulders = build_shoulder_profile(shoulder_df)

    apply_shoulder_profile(lane_sections, shoulders, defaults={"shoulder_width_m": 0.5})

    left_lanes = [lane for lane in lane_sections[0]["left"] if lane["type"] == "shoulder"]
    right_lanes = [lane for lane in lane_sections[0]["right"] if lane["type"] == "shoulder"]

    assert left_lanes and left_lanes[0]["width"] == 2.0
    assert right_lanes and right_lanes[0]["width"] == 1.5

    next_left = [lane for lane in lane_sections[1]["left"] if lane["type"] == "shoulder"][0]
    left_shoulder_id = next_left["id"]
    assert left_shoulder_id not in {lane["id"] for lane in lane_sections[0]["left"] if lane["type"] != "shoulder"}
    assert next_left["predecessors"] == [left_shoulder_id]
    assert left_lanes[0]["successors"] == [left_shoulder_id]
