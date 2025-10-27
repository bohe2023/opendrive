import csv
from typing import Iterable

from csv2xodr.simpletable import DataFrame

def _read_with_encoding(path: str, encoding: str) -> DataFrame:
    with open(path, encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
    fieldnames: Iterable[str] = reader.fieldnames or []
    return DataFrame(rows, columns=list(fieldnames))


def read_csv_any(path, encodings=("utf-8", "cp932", "gb18030")) -> DataFrame:
    """複数のエンコーディングを順に試しCSV読み込みの成功率を高める。"""
    last_err = None
    for enc in encodings:
        try:
            return _read_with_encoding(path, enc)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"CSVを読み込めませんでした: {path}")
