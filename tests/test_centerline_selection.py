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
