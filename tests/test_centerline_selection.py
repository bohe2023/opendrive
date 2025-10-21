import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from csv2xodr.normalize.core import (
    build_centerline,
    filter_dataframe_by_path,
    latlon_to_local_xy,
    select_best_path_id,
)


def test_build_centerline_chooses_longest_path():
    path_a = [
        (35.0, 135.0),
        (35.0, 135.0001),
        (35.0, 135.0002),
        (35.0, 135.0003),
    ]
    path_b = [
        (35.0005, 135.0),
        (35.0005, 135.00005),
        (35.0005, 135.0001),
    ]

    rows = []
    for idx in range(max(len(path_a), len(path_b))):
        if idx < len(path_a):
            rows.append(("A",) + path_a[idx])
        if idx < len(path_b):
            rows.append(("B",) + path_b[idx])

    df = pd.DataFrame(rows, columns=["Path Id", "緯度[deg]", "経度[deg]"])

    center, _ = build_centerline(df, None)

    # Only the four points from path A should remain.
    assert len(center) == len(path_a)

    lat0 = np.mean([lat for lat, _ in path_a])
    lon0 = np.mean([lon for _, lon in path_a])
    ax, ay = latlon_to_local_xy(
        np.array([lat for lat, _ in path_a]),
        np.array([lon for _, lon in path_a]),
        lat0,
        lon0,
    )
    expected_length = float(np.hypot(np.diff(ax), np.diff(ay)).sum())

    assert center["s"].iloc[-1] == pytest.approx(expected_length, rel=1e-6)


def test_select_best_path_id_matches_longest_polyline():
    rows = [
        ("A", 35.0, 135.0),
        ("A", 35.0, 135.0002),
        ("B", 35.0001, 135.0),
    ]
    df = pd.DataFrame(rows, columns=["Path Id", "緯度[deg]", "経度[deg]"])

    assert select_best_path_id(df) == "A"


def test_filter_dataframe_by_path_restricts_rows():
    df = pd.DataFrame(
        {
            "Path Id": ["A", "B", "A"],
            "緯度[deg]": [35.0, 36.0, 35.1],
            "経度[deg]": [135.0, 136.0, 135.1],
        }
    )

    filtered = filter_dataframe_by_path(df, "A")
    assert len(filtered) == 2
    assert set(filtered["Path Id"].tolist()) == {"A"}


def test_build_centerline_averages_duplicate_offsets():
    df = pd.DataFrame(
        {
            "Path Id": ["A", "A", "A", "A"],
            "Offset[cm]": [0, 0, 100, 100],
            "緯度[deg]": [35.0, 35.000001, 35.000002, 35.000003],
            "経度[deg]": [139.0, 139.000001, 139.000002, 139.000003],
        }
    )

    center, _ = build_centerline(df, None)

    # Only unique offsets should remain.
    assert len(center) == 2

    # Offsets are reported in centimetres, so "s" should use metres.
    assert center["s"].to_list() == pytest.approx([0.0, 1.0], rel=1e-6)

    lat_avg0 = np.mean([35.0, 35.000001])
    lon_avg0 = np.mean([139.0, 139.000001])
    lat_avg1 = np.mean([35.000002, 35.000003])
    lon_avg1 = np.mean([139.000002, 139.000003])

    xs, ys = latlon_to_local_xy(
        np.array([lat_avg0, lat_avg1]),
        np.array([lon_avg0, lon_avg1]),
        np.mean([lat_avg0, lat_avg1]),
        np.mean([lon_avg0, lon_avg1]),
    )

    assert center["x"].to_list() == pytest.approx(xs.tolist(), rel=1e-6)
    assert center["y"].to_list() == pytest.approx(ys.tolist(), rel=1e-6)


def test_build_centerline_discards_repeated_samples():
    df = pd.DataFrame(
        {
            "Path Id": ["A"] * 5,
            "Offset[cm]": [0, 0, 100, 200, 200],
            "緯度[deg]": [35.0, 35.0, 35.0005, 35.001, 35.001],
            "経度[deg]": [139.0, 139.0, 139.0005, 139.001, 139.001],
        }
    )

    center, _ = build_centerline(df, None)

    # Duplicate offsets should be collapsed so that the resulting ``s`` values
    # remain strictly increasing, preventing zero-length geometry segments.
    assert center["s"].to_list() == pytest.approx([0.0, 1.0, 2.0], rel=1e-6)


def test_build_centerline_prefers_matching_base_point():
    df_geo = pd.DataFrame(
        {
            "Path Id": ["B", "B", "B"],
            "Offset[cm]": [0, 50, 100],
            "緯度[deg]": [35.0, 35.0, 35.0005],
            "経度[deg]": [135.0, 135.0001, 135.0002],
        }
    )

    df_base = pd.DataFrame(
        {
            "Path Id": ["A", "B"],
            "Offset[cm]": [0, 0],
            "End Offset[cm]": [100, 100],
            "緯度[deg]": [10.0, 20.0],
            "経度[deg]": [30.0, 40.0],
        }
    )

    _, origin = build_centerline(df_geo, df_base)

    assert origin[0] == pytest.approx(20.0)
    assert origin[1] == pytest.approx(40.0)
