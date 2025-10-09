from __future__ import annotations

from csv import DictReader
from io import StringIO

from pythonProject.add_shape_index import assign_shape_indices, add_shape_index_column, SHAPE_INDEX_COLUMN


def test_assign_shape_indices_cycles_within_lane():
    rows = [
        {"Lane ID": "A", "形状要素点数": "3"},
        {"Lane ID": "A", "形状要素点数": "3"},
        {"Lane ID": "A", "形状要素点数": "3"},
        {"Lane ID": "A", "形状要素点数": "3"},
        {"Lane ID": "A", "形状要素点数": "3"},
    ]

    result = assign_shape_indices(rows)
    indices = [int(row[SHAPE_INDEX_COLUMN]) for row in result]

    assert indices == [0, 1, 2, 0, 1]


def test_assign_shape_indices_handles_missing_counts():
    rows = [
        {"Lane ID": "X", "形状要素点数": ""},
        {"Lane ID": "X", "形状要素点数": "0"},
        {"Lane ID": "X", "形状要素点数": "not-a-number"},
    ]

    result = assign_shape_indices(rows)
    indices = [int(row[SHAPE_INDEX_COLUMN]) for row in result]

    assert indices == [0, 1, 2]


def test_add_shape_index_column_roundtrip(tmp_path):
    content = """Lane ID,形状要素点数,緯度[deg]\nA,2,0.0\nA,2,1.0\nA,2,2.0\n"""
    source = tmp_path / "geometry.csv"
    source.write_text(content, encoding="cp932")

    add_shape_index_column(source)

    updated = source.read_text(encoding="cp932")
    reader = DictReader(StringIO(updated))
    rows = list(reader)

    assert reader.fieldnames[-1] == SHAPE_INDEX_COLUMN
    assert [row[SHAPE_INDEX_COLUMN] for row in rows] == ["0", "1", "0"]
