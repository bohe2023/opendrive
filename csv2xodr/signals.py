"""Helpers for converting sign CSV tables into OpenDRIVE <signal> entries."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from csv2xodr.normalize.core import _find_column, _to_float, latlon_to_local_xy
from csv2xodr.simpletable import DataFrame
from csv2xodr.line_geometry import _CenterlineProjector


@dataclass
class SignalExport:
    """Container bundling exported signals with their support objects."""

    signals: List[Dict[str, Any]]
    objects: List[Dict[str, Any]]


_SPEED_SIGN_TYPE_CODE = "1000011"
_SIGN_COUNTRY = "OpenDRIVE"
_SIGN_BOARD_HEIGHT_M = 0.75
_SIGN_BOARD_WIDTH_M = 0.55
_SIGN_BOARD_Z_OFFSET_M = 1.5
_SUPPORT_OBJECT_TYPE = "pole"
_SUPPORT_LENGTH_M = 0.10
_SUPPORT_WIDTH_M = 0.10
_SUPPORT_HEIGHT_M = 2.0
_SIGN_STACK_GAP_M = 0.15


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


def _extract_speed_hint(text: str) -> Optional[float]:
    """Best-effort extraction of a numeric speed from supplementary text.

    Some Japanese datasets mark variable/digital speed limit signs with a
    supplementary classification value instead of filling ``最高速度値``.
    These hints are typically a single digit (e.g. ``"4"`` for "40km/h").
    When such a hint is present we try to recover a plausible km/h value so
    that the exported OpenDRIVE sign carries a non-zero ``value`` and can be
    rendered by viewers.
    """

    if not text:
        return None

    # Normalise full-width digits into ASCII and strip out everything that is
    # not a decimal number.  ``unicodedata`` keeps the dependency footprint
    # small and copes with values such as "４".
    normalised = unicodedata.normalize("NFKC", text)
    digits = "".join(ch for ch in normalised if ch.isdigit())
    if not digits:
        return None

    try:
        value = int(digits)
    except ValueError:  # pragma: no cover - defensive, should not happen
        return None

    if len(digits) == 1:
        value *= 10

    if value <= 0 or value > 200:
        # Filter obvious sentinels such as 65535 or corrupted readings.
        return None

    return float(value)


def _normalise_height(raw_value: Any, column_name: Optional[str]) -> Optional[float]:
    """Convert height readings to metres while filtering obvious sentinels."""

    height = _to_float(raw_value)
    if height is None:
        return None
    if not math.isfinite(height):
        return None
    if abs(height) >= 1.0e4:
        # Values such as 65535 are commonly used as "unknown" sentinels.
        return None

    column_hint = column_name or ""
    lowered = column_hint.lower()
    if "[cm]" in column_hint or "cm" in lowered:
        return height * 0.01
    if abs(height) > 50.0:
        # Fallback heuristic: anything taller than a typical gantry is likely
        # expressed in centimetres even if the column header lacks units.
        return height * 0.01
    return height


def _to_int(value: Any) -> Optional[int]:
    """Best-effort conversion of ``value`` to :class:`int`."""

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 0)
        except ValueError:
            pass

    numeric = _to_float(value)
    if numeric is None:
        return None

    try:
        return int(numeric)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None


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
    centerline: Optional[DataFrame] = None,
    geo_origin: Optional[Tuple[float, float]] = None,
) -> SignalExport:
    """Generate OpenDRIVE signal dictionaries from a sign profile table."""

    signals: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    if df_sign is None or len(df_sign) == 0:
        if log_fn is not None:
            source = sign_filename or "<missing>"
            log_fn(f"[signals] 0 entries from {source}")
        return SignalExport(signals=signals, objects=objects)

    offset_col = _find_column(df_sign, "offset")
    if offset_col is None:
        if log_fn is not None:
            source = sign_filename or "<missing>"
            log_fn(f"[signals] 0 entries from {source} (no offset column)")
        return SignalExport(signals=signals, objects=objects)

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
        return SignalExport(signals=signals, objects=objects)

    _, offset_to_metres = _normalise_offset(offsets_cm)

    entries: List[Tuple[float, Dict[str, Any]]] = []
    upper_country = (country or "").upper() or "JPN"

    speed_col_jp = None
    supplementary_col = None
    digital_col = None
    attribute_flag_col = None
    type_code_col_jp = None
    speed_col_us = None
    shape_col = None
    type_col = None
    height_col = None

    arrangement_col = None
    instance_id_col = None

    if upper_country == "JPN":
        speed_col_jp = (
            _find_column(df_sign, "最高速度情報")
            or _find_column(df_sign, "最高速度値")
            or _find_column(df_sign, "最高速度")
        )
        supplementary_col = _find_column(df_sign, "補助標識")
        digital_col = _find_column(df_sign, "digital")
        attribute_flag_col = _find_column(df_sign, "標識付加属性")
        type_code_col_jp = _find_column(df_sign, "標識情報種別")
        height_col = (
            _find_column(df_sign, "地上高さ")
            or _find_column(df_sign, "地上高")
            or _find_column(df_sign, "height", "ground")
        )
    else:
        speed_col_us = _find_column(df_sign, "speed", "limit")
        shape_col = _find_column(df_sign, "shape")
        digital_col = _find_column(df_sign, "digital")
        type_col = _find_column(df_sign, "type")
        height_col = _find_column(df_sign, "height", exclude=("elevation",))
        width_col = _find_column(df_sign, "width")
        elevation_col = _find_column(df_sign, "elevation")
        azimuth_col = _find_column(df_sign, "azimuth")
        arrangement_col = _find_column(df_sign, "標識情報", "配列")
        instance_id_col = _find_column(df_sign, "instance", "id")
        sign_id_col = (
            _find_column(df_sign, "標識", "id")
            or _find_column(df_sign, "sign", "id")
            or _find_column(df_sign, "sign", "identifier")
        )

    lat_col = (
        _find_column(df_sign, "緯度")
        or _find_column(df_sign, "latitude")
        or _find_column(df_sign, "lat")
    )
    lon_col = (
        _find_column(df_sign, "経度")
        or _find_column(df_sign, "longitude")
        or _find_column(df_sign, "lon")
    )

    x_col = None
    y_col = None
    for col in df_sign.columns:
        lowered = col.strip().lower()
        if x_col is None and any(token in lowered for token in ("x[", "位置x", "pos x", "east")):
            x_col = col
        if y_col is None and any(token in lowered for token in ("y[", "位置y", "pos y", "north")):
            y_col = col

    projector: Optional[_CenterlineProjector] = None
    if centerline is not None:
        try:
            candidate = _CenterlineProjector(centerline)
        except Exception:  # pragma: no cover - defensive guard
            candidate = None
        if candidate is not None and candidate.is_valid:
            projector = candidate

    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    if geo_origin is not None:
        origin_lat, origin_lon = geo_origin

    group_vertical_offsets: Dict[Tuple[str, float], float] = {}

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

        if projector is not None:
            sample_x: Optional[float] = None
            sample_y: Optional[float] = None

            if lat_col is not None and lon_col is not None and origin_lat is not None and origin_lon is not None:
                lat_val = _to_float(row[lat_col])
                lon_val = _to_float(row[lon_col])
                if lat_val is not None and lon_val is not None:
                    try:
                        sample_x, sample_y = latlon_to_local_xy([lat_val], [lon_val], origin_lat, origin_lon)
                        sample_x = float(sample_x[0])
                        sample_y = float(sample_y[0])
                    except Exception:  # pragma: no cover - defensive guard
                        sample_x = None
                        sample_y = None

            if sample_x is None or sample_y is None:
                x_val = _to_float(row[x_col]) if x_col is not None else None
                y_val = _to_float(row[y_col]) if y_col is not None else None
                if x_val is not None and y_val is not None:
                    sample_x = float(x_val)
                    sample_y = float(y_val)

            if sample_x is not None and sample_y is not None:
                projection = projector.project(sample_x, sample_y, approx_s=s_pos)
                if projection is not None:
                    s_pos = float(projection.s)

        if upper_country == "JPN":
            speed_val = _format_numeric(row[speed_col_jp]) if speed_col_jp else None
            is_digital = False
            if digital_col is not None:
                is_digital = _as_bool(row[digital_col])
            if not is_digital and attribute_flag_col is not None:
                flags = _to_int(row[attribute_flag_col])
                if flags is not None and flags & 0x80:
                    is_digital = True
            if not is_digital and type_code_col_jp is not None:
                type_code = _to_int(row[type_code_col_jp])
                if type_code is not None and type_code & 0x80000000:
                    is_digital = True
            if speed_val is not None and abs(speed_val) <= 1e-6:
                is_digital = True
            if speed_val is None and not is_digital:
                continue
            attrs: Dict[str, Any] = {
                "s": s_pos,
                "t": 0.0,
                "type": _SPEED_SIGN_TYPE_CODE,
                "unit": "km/h",
                "country": _SIGN_COUNTRY,
                "dynamic": "yes" if is_digital else "no",
                "orientation": "+",
                "height": _SIGN_BOARD_HEIGHT_M,
                "width": _SIGN_BOARD_WIDTH_M,
                "zOffset": _SIGN_BOARD_Z_OFFSET_M,
            }
            if speed_val is not None and not (is_digital and abs(speed_val) <= 1e-6):
                attrs["value"] = speed_val
            else:
                attrs["value"] = speed_val if speed_val is not None else 0.0
            try:
                subtype_value = float(attrs.get("value", 0.0))
            except (TypeError, ValueError):
                subtype_value = 0.0
            if math.isfinite(subtype_value):
                attrs["subtype"] = str(int(round(subtype_value)))
            else:
                attrs["subtype"] = "0"
            if supplementary_col is not None:
                supplementary = row[supplementary_col]
                text = str(supplementary).strip() if supplementary is not None else ""
                if text and text not in {"65535", "-1"}:
                    attrs["supplementary"] = text
                    if attrs["dynamic"] == "yes" and (
                        speed_val is None or abs(speed_val) <= 1e-6
                    ):
                        hint = _extract_speed_hint(text)
                        if hint is not None:
                            attrs["value"] = hint

            if height_col is not None:
                z_offset = _normalise_height(row[height_col], height_col)
                if z_offset is not None:
                    attrs["zOffset"] = z_offset
            entries.append((s_pos, attrs))
        else:
            speed_val = _format_numeric(row[speed_col_us]) if speed_col_us else None
            subtype = "max"
            if speed_val is None:
                derived, derived_subtype = _lookup_us_speed(row[type_col] if type_col else None)
                speed_val = derived
                if derived_subtype:
                    subtype = derived_subtype
            is_digital = _as_bool(row[digital_col]) if digital_col is not None else False
            raw_type = ""
            attrs: Dict[str, Any] = {
                "s": s_pos,
                "t": 0.0,
                "country": "US",
                "dynamic": "yes" if is_digital else "no",
            }
            if speed_val is not None:
                attrs.update(
                    {
                        "type": "speed",
                        "subtype": subtype,
                        "unit": "mph",
                        "value": speed_val,
                    }
                )
            else:
                raw_type = str(row[type_col]).strip() if type_col is not None else ""
                if raw_type:
                    attrs["subtype"] = raw_type
                else:
                    attrs["subtype"] = "general"
                attrs.update({"type": "information", "value": 0.0})
            if shape_col is not None:
                shape = row[shape_col]
                text = str(shape).strip() if shape is not None else ""
                if text:
                    attrs["shape"] = text
                    if "type" in attrs and attrs["type"] != "speed" and not raw_type:
                        attrs["subtype"] = text
            board_height = None
            if height_col is not None:
                height_val = _format_numeric(row[height_col])
                if height_val is not None and height_val > 0.0:
                    attrs["height"] = height_val
                    board_height = float(height_val)
            if board_height is None:
                board_height = _SIGN_BOARD_HEIGHT_M
                attrs.setdefault("height", board_height)
            if width_col is not None:
                width_val = _format_numeric(row[width_col])
                if width_val is not None and width_val > 0.0:
                    attrs["width"] = width_val
            if elevation_col is not None:
                elevation_val = _format_numeric(row[elevation_col])
                if elevation_val is not None:
                    attrs["pitch"] = math.radians(elevation_val)
            if azimuth_col is not None:
                azimuth_val = _format_numeric(row[azimuth_col])
                if azimuth_val is not None:
                    attrs.setdefault("hOffset", math.radians(azimuth_val))
            arrangement_total = _to_int(row[arrangement_col]) if arrangement_col is not None else None
            base_z_offset = _to_float(attrs.get("zOffset"))
            if base_z_offset is None:
                base_z_offset = _SIGN_BOARD_Z_OFFSET_M
            support_identifier = ""
            if instance_id_col is not None:
                identifier = row[instance_id_col]
                support_identifier = str(identifier).strip() if identifier is not None else ""
            if not support_identifier:
                support_identifier = f"{offset_cm}:{s_pos}"  # fallback key when dataset lacks IDs
            stack_key = (support_identifier, round(float(s_pos), 6))
            next_z_offset = group_vertical_offsets.get(stack_key, float(base_z_offset))
            attrs["zOffset"] = next_z_offset
            gap = _SIGN_STACK_GAP_M if arrangement_total is None or arrangement_total > 1 else 0.0
            group_vertical_offsets[stack_key] = next_z_offset + float(board_height) + gap
            if sign_id_col is not None:
                identifier = row[sign_id_col]
                text = str(identifier).strip() if identifier is not None else ""
                if text:
                    attrs["name"] = text
            entries.append((s_pos, attrs))

    entries.sort(key=lambda item: item[0])
    for idx, (s_pos, attrs) in enumerate(entries, start=1):
        attrs.setdefault("id", f"sig_main_{idx}")
        attrs.setdefault("orientation", "+")
        attrs.setdefault("name", attrs.get("supplementary", ""))
        signals.append(attrs)

        support_attrs: Dict[str, Any] = {
            "id": f"pole_sig_main_{idx}",
            "name": f"sign_pole_{idx}",
            "type": _SUPPORT_OBJECT_TYPE,
            "s": s_pos,
            "t": 0.0,
            "zOffset": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "orientation": "+",
            "length": _SUPPORT_LENGTH_M,
            "width": _SUPPORT_WIDTH_M,
        }
        sign_z_offset = _to_float(attrs.get("zOffset"))
        if sign_z_offset is None:
            sign_z_offset = _SIGN_BOARD_Z_OFFSET_M
        sign_height = _to_float(attrs.get("height"))
        if sign_height is None or sign_height <= 0.0:
            sign_height = _SIGN_BOARD_HEIGHT_M
        support_height = max(_SUPPORT_HEIGHT_M, sign_z_offset + sign_height)
        support_attrs["height"] = support_height
        objects.append(support_attrs)

    if log_fn is not None:
        source = sign_filename or "sign"
        log_fn(f"[signals] {len(signals)} entries from {source}")

    return SignalExport(signals=signals, objects=objects)
