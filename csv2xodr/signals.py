"""Helpers for converting sign CSV tables into OpenDRIVE <signal> entries."""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from csv2xodr.normalize.core import _find_column, _to_float
from csv2xodr.simpletable import DataFrame


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return text in {"1", "true", "yes", "y", "t"}


def _format_numeric(value: Optional[float]) -> Optional[float]:
    """Return a floating point number when *value* contains numeric text."""

    if value is None:
        return None

    # ``_to_float`` already normalises locale dependent formatting but it
    # expects a clean numeric token.  Real-world CSV dumps occasionally embed
    # speed limits inside human-readable strings such as ``"約50km/h"`` or
    # include measurement units in brackets.  Extract the first numeric token
    # before delegating to the shared conversion helper so that we still honour
    # grouping separators and full-width digits.
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
        if match is not None:
            value = match.group(0)

    numeric = _to_float(value)
    if numeric is None:
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _normalise_offset(values: List[float]) -> Tuple[float, Callable[[float], float]]:
    if not values:
        return 0.0, lambda value: float(value)
    origin = min(values)

    def _convert(raw_cm: float) -> float:
        metres = (raw_cm - origin) * 0.01
        return max(0.0, metres)

    return origin, _convert


_US_TYPE_RANGES: List[Tuple[int, int, str]] = [
    (66, 95, "max"),
    (96, 125, "min"),
    (126, 155, "truck"),
    (156, 185, "night"),
    (186, 195, "warning"),
]


def _lookup_us_speed(type_code: Optional[Any]) -> Tuple[Optional[float], Optional[str]]:
    if type_code is None:
        return None, None
    try:
        code = int(float(type_code))
    except (TypeError, ValueError):
        return None, None
    for start, end, category in _US_TYPE_RANGES:
        if start <= code <= end:
            value = 5.0 * (code - start + 1)
            return value, category
    return None, None


def generate_signals(
    df_sign: Optional[DataFrame],
    *,
    country: str,
    offset_mapper: Optional[Callable[[float], float]] = None,
    sign_filename: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> List[Dict[str, Any]]:
    """Generate OpenDRIVE signal dictionaries from a sign profile table."""

    signals: List[Dict[str, Any]] = []
    if df_sign is None or len(df_sign) == 0:
        if log_fn is not None:
            source = sign_filename or "<missing>"
            log_fn(f"[signals] 0 entries from {source}")
        return signals

    offset_col = _find_column(df_sign, "offset")
    if offset_col is None:
        if log_fn is not None:
            source = sign_filename or "<missing>"
            log_fn(f"[signals] 0 entries from {source} (no offset column)")
        return signals

    retrans_col = _find_column(df_sign, "is", "retransmission")

    offsets_cm: List[float] = []
    for idx in range(len(df_sign)):
        row = df_sign.iloc[idx]
        if retrans_col is not None and _as_bool(row[retrans_col]):
            continue
        offset_val = _to_float(row[offset_col])
        if offset_val is not None:
            offsets_cm.append(offset_val)

    if not offsets_cm:
        if log_fn is not None:
            source = sign_filename or "<missing>"
            log_fn(f"[signals] 0 entries from {source} (no valid offsets)")
        return signals

    _, offset_to_metres = _normalise_offset(offsets_cm)

    entries: List[Tuple[float, Dict[str, Any]]] = []
    upper_country = (country or "").upper() or "JPN"

    speed_col_jp = None
    supplementary_col = None
    digital_col = None
    speed_col_us = None
    shape_col = None
    type_col = None

    if upper_country == "JPN":
        speed_col_jp = (
            _find_column(df_sign, "最高速度情報")
            or _find_column(df_sign, "最高速度値")
            or _find_column(df_sign, "最高速度")
        )
        supplementary_col = _find_column(df_sign, "補助標識")
        digital_col = _find_column(df_sign, "digital")
    else:
        speed_col_us = _find_column(df_sign, "speed", "limit")
        shape_col = _find_column(df_sign, "shape")
        digital_col = _find_column(df_sign, "digital")
        type_col = _find_column(df_sign, "type")

    for idx in range(len(df_sign)):
        row = df_sign.iloc[idx]
        if retrans_col is not None and _as_bool(row[retrans_col]):
            continue

        offset_cm = _to_float(row[offset_col])
        if offset_cm is None:
            continue
        offset_m = offset_to_metres(offset_cm)
        if offset_mapper is not None:
            s_pos = float(offset_mapper(offset_m))
        else:
            s_pos = float(offset_m)

        if upper_country == "JPN":
            speed_val = _format_numeric(row[speed_col_jp]) if speed_col_jp else None
            is_digital = False
            if digital_col is not None:
                is_digital = _as_bool(row[digital_col])
            if speed_val is not None and abs(speed_val) <= 1e-6:
                is_digital = True
            if speed_val is None and not is_digital:
                continue
            attrs: Dict[str, Any] = {
                "s": s_pos,
                "t": 0.0,
                "type": "speed",
                "subtype": "max",
                "unit": "km/h",
                "country": "JPN",
                "dynamic": "yes" if is_digital else "no",
            }
            if speed_val is not None and not (is_digital and abs(speed_val) <= 1e-6):
                attrs["value"] = speed_val
            else:
                attrs["value"] = speed_val if speed_val is not None else 0.0
            if supplementary_col is not None:
                supplementary = row[supplementary_col]
                text = str(supplementary).strip() if supplementary is not None else ""
                if text and text not in {"65535", "-1"}:
                    attrs["supplementary"] = text
            entries.append((s_pos, attrs))
        else:
            speed_val = _format_numeric(row[speed_col_us]) if speed_col_us else None
            subtype = "max"
            if speed_val is None:
                derived, derived_subtype = _lookup_us_speed(row[type_col] if type_col else None)
                speed_val = derived
                if derived_subtype:
                    subtype = derived_subtype
            if speed_val is None:
                continue
            is_digital = _as_bool(row[digital_col]) if digital_col is not None else False
            attrs = {
                "s": s_pos,
                "t": 0.0,
                "type": "speed",
                "subtype": subtype,
                "unit": "mph",
                "country": "US",
                "dynamic": "yes" if is_digital else "no",
                "value": speed_val,
            }
            if shape_col is not None:
                shape = row[shape_col]
                text = str(shape).strip() if shape is not None else ""
                if text:
                    attrs["shape"] = text
            entries.append((s_pos, attrs))

    entries.sort(key=lambda item: item[0])
    for idx, (s_pos, attrs) in enumerate(entries, start=1):
        attrs.setdefault("id", f"sig_{idx}")
        attrs.setdefault("orientation", "none")
        attrs.setdefault("name", attrs.get("supplementary", ""))
        signals.append(attrs)

    if log_fn is not None:
        source = sign_filename or "sign"
        log_fn(f"[signals] {len(signals)} entries from {source}")

    return signals
