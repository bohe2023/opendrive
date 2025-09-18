"""A very small subset of pandas-like functionality used by csv2xodr.

This module implements light-weight :class:`DataFrame` and :class:`Series`
objects that support the handful of operations required by the converter.
It intentionally does **not** try to be a drop-in replacement for pandas –
only the methods/behaviours that are exercised inside this project are
implemented.  The goal is to keep the tool functional in minimal
environments where installing heavy third-party dependencies such as
``pandas`` or ``numpy`` is not feasible.

The implementation focuses on:

* column selection via ``df["col"]`` and ``df[["a", "b"]]``
* ``len(df)`` and ``len(series)``
* integer based indexing via ``iloc`` for both series and data frames
* boolean/sequence indexing via ``loc``
* ``filter(like="...")``
* a very small ``groupby`` with ``mean`` aggregation
* helpers such as ``dropna()``, ``unique()``, ``duplicated()`` and
  ``astype(float)`` on series

Only the behaviour that is explicitly relied upon by the rest of the
code base is implemented here.  The API is intentionally tiny but it is
documented and unit tested through the higher-level features of the
converter.  When new functionality is required it can be extended in a
controlled manner.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


def _is_na(value: Any) -> bool:
    """Return ``True`` when *value* should be treated as missing/NaN."""

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "nan":
            return True
    return False


def notna(value: Any) -> bool:
    """Inverse of :func:`_is_na` to mimic ``pandas.notna``."""

    return not _is_na(value)


class Series:
    """A light-weight sequence with a pandas-like interface."""

    def __init__(
        self,
        data: Iterable[Any],
        *,
        index: Optional[Sequence[Any]] = None,
        name: Optional[str] = None,
        kind: str = "column",
    ) -> None:
        self._data: List[Any] = list(data)
        self.name = name
        self._kind = kind
        if index is None:
            if kind == "column":
                self.index: List[Any] = list(range(len(self._data)))
            else:
                self.index = [None for _ in self._data]
        else:
            self.index = list(index)

    # ------------------------------------------------------------------
    # Representation helpers
    def __len__(self) -> int:  # pragma: no cover - behaviour validated indirectly
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __getitem__(self, key: Union[int, str]) -> Any:
        if self._kind == "row" and isinstance(key, str):
            try:
                idx = self.index.index(key)
            except ValueError as exc:  # pragma: no cover - defensive programming
                raise KeyError(key) from exc
            return self._data[idx]
        return self._data[key]

    @property
    def values(self) -> List[Any]:
        return list(self._data)

    def to_list(self) -> List[Any]:
        return list(self._data)

    def to_numpy(self) -> List[Any]:  # compatibility shim
        return list(self._data)

    def astype(self, dtype) -> "Series":
        converted: List[Any] = []
        for value in self._data:
            if _is_na(value):
                if dtype is float:
                    converted.append(math.nan)
                else:
                    converted.append(dtype())
                continue
            converted.append(dtype(value))
        return Series(converted, index=self.index, name=self.name, kind=self._kind)

    def duplicated(self) -> "Series":
        seen: set = set()
        flags: List[bool] = []
        for value in self._data:
            if _is_na(value):
                flags.append(False)
                continue
            if value in seen:
                flags.append(True)
            else:
                seen.add(value)
                flags.append(False)
        return Series(flags, index=self.index, kind=self._kind)

    def any(self) -> bool:
        return any(bool(v) for v in self._data)

    def dropna(self) -> "Series":
        data: List[Any] = []
        new_index: List[Any] = []
        for idx, value in zip(self.index, self._data):
            if _is_na(value):
                continue
            data.append(value)
            new_index.append(idx)
        return Series(data, index=new_index, name=self.name, kind=self._kind)

    def unique(self) -> List[Any]:
        seen: List[Any] = []
        for value in self._data:
            if _is_na(value):
                continue
            if value not in seen:
                seen.append(value)
        return seen

    def nunique(self, dropna: bool = True) -> int:
        if dropna:
            return len(self.dropna().unique())
        return len(Series(self._data, index=self.index).unique())

    def iloc(self, idx: int) -> Any:  # pragma: no cover - retained for backward compat
        return self._data[idx]

    # pandas-like attribute
    @property
    def iloc(self) -> "_SeriesILoc":  # type: ignore[override]
        return _SeriesILoc(self)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Series({self._data})"


class _SeriesILoc:
    def __init__(self, series: Series) -> None:
        self._series = series

    def __getitem__(self, item: Union[int, slice]) -> Union[Any, Series]:
        if isinstance(item, slice):
            return Series(
                self._series._data[item],
                index=self._series.index[item],
                name=self._series.name,
                kind=self._series._kind,
            )
        return self._series._data[item]


class DataFrame:
    """A tabular structure that mimics a very small portion of pandas."""

    def __init__(self, data: Union[Dict[str, Iterable[Any]], Iterable[Dict[str, Any]]], *, columns: Optional[List[str]] = None) -> None:
        if isinstance(data, dict):
            cols = list(data.keys())
            length = 0
            for col in cols:
                values = list(data[col])
                data[col] = values
                length = len(values)
            rows: List[Dict[str, Any]] = []
            for i in range(length):
                row = {col: data[col][i] for col in cols}
                rows.append(row)
            self._rows = rows
            self.columns = cols
        else:
            rows = [dict(row) for row in data]
            self._rows = rows
            if columns is not None:
                self.columns = list(columns)
            elif rows:
                self.columns = list(rows[0].keys())
            else:
                self.columns = []

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, item: Union[str, List[str]]) -> Union[Series, "DataFrame"]:
        if isinstance(item, str):
            return Series([row.get(item) for row in self._rows], name=item, kind="column")
        elif isinstance(item, list):
            subset = [{col: row.get(col) for col in item} for row in self._rows]
            return DataFrame(subset, columns=item)
        else:  # pragma: no cover - defensive programming
            raise TypeError("Unsupported key type")

    # pandas-like indexers -------------------------------------------------
    @property
    def iloc(self) -> "_DataFrameILoc":
        return _DataFrameILoc(self)

    @property
    def loc(self) -> "_DataFrameLoc":
        return _DataFrameLoc(self)

    # ------------------------------------------------------------------
    def filter(self, like: Optional[str] = None) -> "DataFrame":
        if like is None:
            return DataFrame(self._rows, columns=self.columns)
        cols = [c for c in self.columns if like in c]
        return self[cols]

    def groupby(self, column: str, sort: bool = True) -> "_GroupBy":
        return _GroupBy(self, column, sort=sort)

    def reset_index(self, drop: bool = False) -> "DataFrame":
        if drop:
            return DataFrame(self._rows, columns=self.columns)
        new_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(self._rows):
            new_row = dict(row)
            new_row["index"] = idx
            new_rows.append(new_row)
        cols = ["index"] + [c for c in self.columns if c != "index"]
        return DataFrame(new_rows, columns=cols)

    def to_dicts(self) -> List[Dict[str, Any]]:  # pragma: no cover - debug helper
        return [dict(row) for row in self._rows]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"DataFrame(rows={len(self)}, columns={self.columns})"


class _DataFrameILoc:
    def __init__(self, df: DataFrame) -> None:
        self._df = df

    def _resolve_row(self, selector: Union[int, slice, List[int]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        rows = self._df._rows
        if isinstance(selector, int):
            return rows[selector]
        if isinstance(selector, slice):
            return rows[selector]
        if isinstance(selector, list):
            return [rows[i] for i in selector]
        raise TypeError("Unsupported iloc row selector")

    def _resolve_columns(self, selector: Union[int, slice, List[int]]) -> List[str]:
        cols = self._df.columns
        if isinstance(selector, int):
            return [cols[selector]]
        if isinstance(selector, slice):
            return cols[selector]
        if isinstance(selector, list):
            return [cols[i] for i in selector]
        raise TypeError("Unsupported iloc column selector")

    def __getitem__(self, item: Union[int, slice, Tuple[Union[int, slice, List[int]], Union[int, slice, List[int]]]]):
        if isinstance(item, tuple):
            row_sel, col_sel = item
            rows = self._resolve_row(row_sel)
            cols = self._resolve_columns(col_sel)
            if isinstance(rows, list):
                subset = [{col: row.get(col) for col in cols} for row in rows]
                if len(subset) == 1 and len(cols) == 1:
                    return subset[0][cols[0]]
                return DataFrame(subset, columns=cols)
            else:
                row = rows
                if len(cols) == 1:
                    return row.get(cols[0])
                return Series([row.get(col) for col in cols], index=cols, kind="row")

        rows = self._resolve_row(item)
        if isinstance(rows, list):
            return DataFrame(rows, columns=self._df.columns)
        return Series([rows.get(col) for col in self._df.columns], index=self._df.columns, kind="row")


class _DataFrameLoc:
    def __init__(self, df: DataFrame) -> None:
        self._df = df

    def __getitem__(self, item: Union[slice, List[bool], List[int]]):
        rows = self._df._rows
        if isinstance(item, slice):
            return DataFrame(rows[item], columns=self._df.columns)
        if isinstance(item, list):
            if all(isinstance(v, bool) for v in item):
                selected = [row for row, keep in zip(rows, item) if keep]
                return DataFrame(selected, columns=self._df.columns)
            if all(isinstance(v, int) for v in item):
                selected = [rows[i] for i in item]
                return DataFrame(selected, columns=self._df.columns)
        raise TypeError("Unsupported loc selector")


class _GroupBy:
    def __init__(self, df: DataFrame, column: str, *, sort: bool = True) -> None:
        self._df = df
        self._column = column
        self._sort = sort

    def __getitem__(self, columns: List[str]) -> "_GroupBySelection":
        return _GroupBySelection(self._df, self._column, columns, self._sort)


class _GroupBySelection:
    def __init__(self, df: DataFrame, group_col: str, columns: List[str], sort: bool) -> None:
        self._df = df
        self._group_col = group_col
        self._columns = columns
        self._sort = sort

    def mean(self) -> DataFrame:
        sums: Dict[Any, Dict[str, float]] = {}
        counts: Dict[Any, Dict[str, int]] = {}
        for row in self._df._rows:
            key = row.get(self._group_col)
            if _is_na(key):
                continue
            if key not in sums:
                sums[key] = {col: 0.0 for col in self._columns}
                counts[key] = {col: 0 for col in self._columns}
            for col in self._columns:
                value = row.get(col)
                if _is_na(value):
                    continue
                try:
                    fval = float(value)
                except Exception:  # pragma: no cover - mirrors pandas behaviour
                    continue
                sums[key][col] += fval
                counts[key][col] += 1

        keys = list(sums.keys())
        if self._sort:
            try:
                keys.sort()
            except TypeError:  # pragma: no cover - fall back to insertion order
                pass

        out_rows: List[Dict[str, Any]] = []
        for key in keys:
            row: Dict[str, Any] = {self._group_col: key}
            for col in self._columns:
                count = counts[key][col]
                if count == 0:
                    row[col] = None
                else:
                    row[col] = sums[key][col] / count
            out_rows.append(row)
        return DataFrame(out_rows, columns=[self._group_col] + self._columns)


__all__ = [
    "DataFrame",
    "Series",
    "notna",
]

