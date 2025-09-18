import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from csv2xodr.normalize.core import build_centerline, latlon_to_local_xy


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


def test_build_centerline_chooses_longest_line_feature():
    shared_path = "A"
    feature_long = "F1"
    feature_short = "F2"

    long_points = [
        (35.0, 135.0),
        (35.0001, 135.00005),
        (35.0002, 135.0001),
    ]
    short_points = [
        (35.1, 135.1),
        (35.10005, 135.10005),
    ]

    rows = []
    for idx in range(max(len(long_points), len(short_points))):
        if idx < len(long_points):
            rows.append((shared_path, feature_long) + long_points[idx])
        if idx < len(short_points):
            rows.append((shared_path, feature_short) + short_points[idx])

    df = pd.DataFrame(
        rows,
        columns=["Path Id", "ライン型地物ID", "緯度[deg]", "経度[deg]"],
    )

    center, _ = build_centerline(df, None)

    assert len(center) == len(long_points)
