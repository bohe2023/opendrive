from csv2xodr.normalize.core import (
    build_curvature_profile,
    build_elevation_profile_from_slopes,
    build_geometry_segments,
    build_shoulder_profile,
    build_slope_profile,
    build_superelevation_profile,
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
    center = DataFrame({
        "s": [0.0, 1.0, 2.0],
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
        "hdg": [0.0, 0.0, 0.0],
    })

    curvature_df = DataFrame(
        {
            "Offset[cm]": [0, 100],
            "End Offset[cm]": [100, 200],
            "曲率値[rad/m]": [0.0, 0.1],
            "Is Retransmission": ["False", "False"],
        }
    )

    curvature_segments = build_curvature_profile(curvature_df)
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

    curvature_segments = build_curvature_profile(curvature_df)

    strict_geometry = build_geometry_segments(
        center,
        curvature_segments,
        max_endpoint_deviation=0.05,
    )

    assert strict_geometry == []

    relaxed_geometry = build_geometry_segments(
        center,
        curvature_segments,
        max_endpoint_deviation=0.5,
    )

    assert len(relaxed_geometry) == 1

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
