"""Utilities for parsing white line geometry from CSV inputs."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from csv2xodr.normalize.core import _smooth_series, latlon_to_local_xy
from csv2xodr.simpletable import DataFrame
from csv2xodr.topology.core import _canonical_numeric


def _find_column(columns: Iterable[str], *keywords: str) -> Optional[str]:
    lowered = [kw.lower() for kw in keywords]
    for col in columns:
        stripped = col.strip()
        value = stripped.lower()
        if all(keyword in value for keyword in lowered):
            return col
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return None
        return float(text)
    except Exception:  # pragma: no cover - defensive
        return None


def _geometry_signature(points: List[Tuple[float, float, float, float]]) -> Tuple[Tuple[int, int, int, int], ...]:
    return tuple(
        (
            int(round(s * 1000)),
            int(round(x * 1000)),
            int(round(y * 1000)),
            int(round(z * 1000)),
        )
        for s, x, y, z in points
    )


def _normalise_key(value: Any) -> Optional[str]:
    canonical = _canonical_numeric(value, allow_negative=True)
    if canonical is not None:
        return canonical
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_line_geometry_lookup(
    line_geom_df: Optional[DataFrame],
    *,
    offset_mapper=None,
    lat0: float,
    lon0: float,
    curvature_samples: Optional[Iterable[Dict[str, Any]]] = None,
    centerline: Optional[DataFrame] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return a lookup of canonical line IDs to polylines in local XY."""

    if line_geom_df is None or len(line_geom_df) == 0:
        return {}

    cols = list(line_geom_df.columns)

    line_id_col = (
        _find_column(cols, "ライン", "ID")
        or _find_column(cols, "区画線", "ID")
        or _find_column(cols, "lane", "line", "id")
    )
    lat_col = _find_column(cols, "緯度") or _find_column(cols, "Latitude")
    lon_col = _find_column(cols, "経度") or _find_column(cols, "Longitude")
    z_col = _find_column(cols, "高さ") or _find_column(cols, "標高") or _find_column(cols, "Height")
    offset_col = None
    for col in cols:
        stripped = col.strip().lower()
        if "offset" in stripped and "end" not in stripped:
            offset_col = col
            break
    end_offset_col = _find_column(cols, "End", "Offset")
    type_col = _find_column(cols, "Type") or _find_column(cols, "種別")
    logtime_col = _find_column(cols, "ログ") or _find_column(cols, "log", "Time")
    instance_col = _find_column(cols, "Instance") or _find_column(cols, "インスタンス")
    flag_col = _find_column(cols, "3D") or _find_column(cols, "属性")
    is_retrans_col = _find_column(cols, "Is", "Retransmission")
    shape_count_col = (
        _find_column(cols, "形状", "要素")
        or _find_column(cols, "shape", "count")
        or _find_column(cols, "shape", "points")
    )
    path_col = _find_column(cols, "Path", "Id")
    lane_col = _find_column(cols, "Lane", "Number") or _find_column(cols, "Lane", "No")

    if line_id_col is None or lat_col is None or lon_col is None:
        return {}

    grouped: Dict[Tuple[str, Any, Any, Any, Any], Dict[str, Any]] = {}
    segment_progress: Dict[
        Tuple[Any, Any, Any, Any, Any, float, Optional[float]], Dict[str, float]
    ] = {}

    # ``curvature_samples`` had previously been used to inject raw curvature
    # measurements.  The new smoothing-based pipeline derives curvature from the
    # lane geometry directly, therefore any supplied samples are ignored.  The
    # parameter is kept for backwards compatibility with existing call sites.

    for idx in range(len(line_geom_df)):
        row = line_geom_df.iloc[idx]

        line_id = _canonical_numeric(row[line_id_col], allow_negative=True)
        if line_id is None:
            continue

        lat_val = _to_float(row[lat_col])
        lon_val = _to_float(row[lon_col])
        if lat_val is None or lon_val is None:
            continue

        group_key = (
            line_id,
            row[logtime_col] if logtime_col else None,
            row[instance_col] if instance_col else None,
            row[flag_col] if flag_col else None,
            row[type_col] if type_col else None,
        )

        entry = grouped.setdefault(
            group_key,
            {
                "line_id": line_id,
                "lat": [],
                "lon": [],
                "z": [],
                "offset": [],
                "shape_index": [],
                "has_true": False,
                "has_false": False,
                "has_flag": False,
                "path_token": None,
                "lane_token": None,
            },
        )

        if path_col is not None and entry.get("path_token") is None:
            entry["path_token"] = _normalise_key(row[path_col])

        if lane_col is not None and entry.get("lane_token") is None:
            entry["lane_token"] = _normalise_key(row[lane_col])

        z_val = _to_float(row[z_col]) if z_col else None
        if z_val is not None:
            if not math.isfinite(z_val) or abs(z_val) >= 1e4:
                z_val = None
        if z_val is None:
            existing = entry.get("z")
            if existing:
                z_val = existing[-1]
            else:
                z_val = 0.0

        off_cm_raw = None
        if offset_col is not None:
            off_cm_raw = _to_float(row[offset_col])
        if off_cm_raw is None and end_offset_col is not None:
            off_cm_raw = _to_float(row[end_offset_col])

        if off_cm_raw is None:
            continue

        end_cm_val = _to_float(row[end_offset_col]) if end_offset_col is not None else None

        segment_key = (
            line_id,
            row[logtime_col] if logtime_col else None,
            row[instance_col] if instance_col else None,
            row[flag_col] if flag_col else None,
            row[type_col] if type_col else None,
            float(off_cm_raw),
            float(end_cm_val) if end_cm_val is not None else None,
        )

        tracker = segment_progress.get(segment_key)
        if tracker is None:
            count_val: Optional[int] = None
            if shape_count_col is not None:
                try:
                    raw_value = row[shape_count_col]
                except Exception:  # pragma: no cover - defensive
                    raw_value = None
                raw_count = _to_float(raw_value)
                if raw_count is not None and raw_count > 0:
                    try:
                        count_val = int(round(raw_count))
                    except Exception:  # pragma: no cover - defensive
                        count_val = None
            step_val: Optional[float] = None
            if count_val is not None and count_val > 1 and end_cm_val is not None:
                step_val = (float(end_cm_val) - float(off_cm_raw)) / float(count_val - 1)
            tracker = {
                "index": 0,
                "count": count_val,
                "step": step_val,
                "start": float(off_cm_raw),
            }
            segment_progress[segment_key] = tracker

        index = int(tracker.get("index", 0))
        tracker["index"] = index + 1
        count_val = tracker.get("count")
        step_val = tracker.get("step")
        start_cm = tracker.get("start", float(off_cm_raw))

        if count_val is not None and tracker["index"] >= count_val:
            segment_progress.pop(segment_key, None)

        if step_val is not None and count_val is not None and count_val > 1:
            off_val = start_cm + step_val * index
        else:
            off_val = float(off_cm_raw)

        retrans_flag = None
        if is_retrans_col is not None:
            entry["has_flag"] = True
            value = row[is_retrans_col]
            if isinstance(value, bool):
                retrans_flag = value
            else:
                retrans_flag = str(value).strip().lower() == "true"
        if retrans_flag is True:
            entry["has_true"] = True
        elif retrans_flag is False:
            entry["has_false"] = True

        off_m = off_val / 100.0

        entry["lat"].append(lat_val)
        entry["lon"].append(lon_val)
        entry["z"].append(z_val)
        entry["offset"].append(off_m)
        entry["shape_index"].append(float(index))

    if not grouped:
        return {}

    lookup: Dict[str, List[Dict[str, Any]]] = {}

    global_base_offset: Optional[float] = None
    projector = _CenterlineProjector(centerline) if centerline is not None else None

    for entry in grouped.values():
        for raw in entry.get("offset", []) or []:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if global_base_offset is None or value < global_base_offset:
                global_base_offset = value

    for entry in grouped.values():
        has_true = entry.get("has_true", False)
        has_false = entry.get("has_false", False)
        has_flag = entry.get("has_flag", False)
        if has_flag and has_true and not has_false:
            continue

        lat_vals = entry.get("lat", []) or []
        lon_vals = entry.get("lon", []) or []
        z_vals = entry.get("z", []) or []
        offsets_raw = entry.get("offset", []) or []
        shape_indices = entry.get("shape_index", []) or []

        if not lat_vals or not lon_vals or not z_vals or not offsets_raw:
            continue

        try:
            x_vals, y_vals = latlon_to_local_xy(lat_vals, lon_vals, lat0, lon0)
        except Exception:  # pragma: no cover - defensive
            continue

        if len(x_vals) != len(z_vals) or len(x_vals) != len(offsets_raw):
            continue

        if len(shape_indices) != len(offsets_raw):
            if len(shape_indices) < len(offsets_raw):
                shape_indices = shape_indices + [None] * (len(offsets_raw) - len(shape_indices))
            else:
                shape_indices = shape_indices[: len(offsets_raw)]

        # The Japanese datasets reuse the same ``line_id`` across multiple
        # carriageway segments.  Each time the offset restarts from zero the
        # geometry would jump back to the beginning of the alignment, producing
        # spaghetti-like lane lines in the viewer.  Split the sequence whenever
        # the longitudinal coordinate decreases noticeably so every emitted
        # polyline remains strictly monotonic along ``s``.
        sequences: List[List[Tuple[float, float, float, float, Optional[float], Optional[float]]]] = []
        current: List[Tuple[float, float, float, float, Optional[float], Optional[float]]] = []
        last_s: Optional[float] = None
        reset_threshold = 1e-4  # tolerate sub-millimetre jitter while catching real resets

        entry_base_offset: Optional[float] = None
        for raw in offsets_raw:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if entry_base_offset is None or value < entry_base_offset:
                entry_base_offset = value

        for idx_point, (raw_offset, x_val, y_val, z_val) in enumerate(
            zip(offsets_raw, x_vals, y_vals, z_vals)
        ):
            try:
                offset_m = float(raw_offset)
            except (TypeError, ValueError):
                continue

            absolute_offset = offset_m

            if entry_base_offset is not None and math.isfinite(entry_base_offset):
                if (
                    global_base_offset is not None
                    and math.isfinite(global_base_offset)
                    and entry_base_offset > global_base_offset + 1e-6
                ):
                    offset_m -= global_base_offset
                else:
                    offset_m -= entry_base_offset
            elif global_base_offset is not None and math.isfinite(global_base_offset):
                offset_m -= global_base_offset

            if offset_mapper is not None:
                try:
                    s_float = float(offset_mapper(offset_m))
                except Exception:  # pragma: no cover - defensive
                    continue
            else:
                s_float = float(offset_m)

            if not math.isfinite(s_float):
                continue

            try:
                x_float = float(x_val)
                y_float = float(y_val)
                z_float = float(z_val)
            except (TypeError, ValueError):
                continue

            if last_s is not None and s_float < last_s - reset_threshold:
                if len(current) >= 2:
                    sequences.append(current)
                current = []
                last_s = None

            if current:
                prev_s, prev_x, prev_y, prev_z, _, _ = current[-1]
                if (
                    abs(s_float - prev_s) <= 1e-9
                    and abs(x_float - prev_x) <= 1e-9
                    and abs(y_float - prev_y) <= 1e-9
                    and abs(z_float - prev_z) <= 1e-9
                ):
                    # Skip duplicate samples that would otherwise collapse into
                    # zero-length segments after splitting.
                    last_s = s_float
                    continue

            shape_idx_val = None
            if idx_point < len(shape_indices):
                shape_idx_val = shape_indices[idx_point]
            current.append((s_float, x_float, y_float, z_float, shape_idx_val, absolute_offset))
            last_s = s_float

        if len(current) >= 2:
            sequences.append(current)

        geoms = lookup.setdefault(entry["line_id"], [])

        for seq in sequences:
            points = list(seq)

            used_centerline = False
            if projector is not None and projector.is_valid:
                reproj_sequences = _reproject_sequence(points, projector)
                for reproj in reproj_sequences:
                    resampled = _reconstruct_geometry_from_centerline(reproj, projector)
                    if resampled is None:
                        continue

                    used_centerline = True
                    resampled_s = [float(val) for val in resampled["s"]]
                    resampled_x = [float(val) for val in resampled["x"]]
                    resampled_y = [float(val) for val in resampled["y"]]
                    resampled_z = [float(val) for val in resampled["z"]]
                    signature = _geometry_signature(
                        list(zip(resampled_s, resampled_x, resampled_y, resampled_z))
                    )
                    if any(
                        _geometry_signature(list(zip(g["s"], g["x"], g["y"], g["z"])))
                        == signature
                        for g in geoms
                    ):
                        continue

                    geom_entry = {
                        "s": resampled_s,
                        "x": resampled_x,
                        "y": resampled_y,
                        "z": resampled_z,
                    }

                    curvature_vals = resampled.get("curvature") or []
                    if curvature_vals:
                        geom_entry["curvature"] = [float(val) for val in curvature_vals]

                    geoms.append(geom_entry)

            if used_centerline:
                continue

            resampled = _resample_sequence_with_polynomial(points)
            if resampled is not None:
                resampled_s = [float(val) for val in resampled["s"]]
                resampled_x = [float(val) for val in resampled["x"]]
                resampled_y = [float(val) for val in resampled["y"]]
                resampled_z = [float(val) for val in resampled["z"]]
                resampled_curvature = [float(val) for val in resampled["curvature"]]
                signature = _geometry_signature(
                    list(zip(resampled_s, resampled_x, resampled_y, resampled_z))
                )
                if any(
                    _geometry_signature(list(zip(g["s"], g["x"], g["y"], g["z"])))
                    == signature
                    for g in geoms
                ):
                    continue

                geom_entry = {
                    "s": resampled_s,
                    "x": resampled_x,
                    "y": resampled_y,
                    "z": resampled_z,
                }
                if any(abs(val) > 1e-12 for val in resampled_curvature):
                    geom_entry["curvature"] = resampled_curvature
                else:
                    geom_entry["curvature"] = [0.0 for _ in resampled_curvature]
                geoms.append(geom_entry)
                continue

            signature = _geometry_signature([(p[0], p[1], p[2], p[3]) for p in points])
            if any(
                _geometry_signature(list(zip(g["s"], g["x"], g["y"], g["z"]))) == signature
                for g in geoms
            ):
                continue

            curvature_vals = _estimate_discrete_curvature(points)

            geom_entry = {
                "s": [p[0] for p in points],
                "x": [p[1] for p in points],
                "y": [p[2] for p in points],
                "z": [p[3] for p in points],
            }

            if curvature_vals:
                geom_entry["curvature"] = curvature_vals

            geoms.append(geom_entry)

    return lookup

RESAMPLE_STEP_METERS = 10.0
CENTERLINE_RESAMPLE_STEP = 3.0
CENTERLINE_SMOOTHING_WINDOW = 4.0


@dataclass
class _ProjectionResult:
    s: float
    x: float
    y: float
    tangent: Tuple[float, float]
    segment_index: int


class _CenterlineProjector:
    """Helper that provides arc-length projections onto the centreline."""

    def __init__(self, centerline: Optional[DataFrame]):
        self.is_valid = False
        self._segments: List[Dict[str, float]] = []
        self._s_vals: List[float] = []

        if centerline is None or len(centerline) < 2:
            return

        try:
            s_raw = [float(v) for v in centerline["s"].to_list()]
            x_raw = [float(v) for v in centerline["x"].to_list()]
            y_raw = [float(v) for v in centerline["y"].to_list()]
        except Exception:  # pragma: no cover - defensive
            return

        if len(s_raw) != len(x_raw) or len(s_raw) != len(y_raw):
            return

        segments: List[Dict[str, float]] = []
        for idx in range(len(s_raw) - 1):
            s0 = s_raw[idx]
            s1 = s_raw[idx + 1]
            x0 = x_raw[idx]
            y0 = y_raw[idx]
            x1 = x_raw[idx + 1]
            y1 = y_raw[idx + 1]
            span = s1 - s0
            dx = x1 - x0
            dy = y1 - y0
            length_sq = dx * dx + dy * dy
            if span <= 0.0 and length_sq <= 1e-12:
                continue
            tangent_len = math.hypot(dx, dy)
            if tangent_len <= 1e-9:
                # Degenerate segment – reuse the previous tangent if available.
                if segments:
                    tangent = (segments[-1]["tx"], segments[-1]["ty"])
                else:
                    tangent = (1.0, 0.0)
            else:
                tangent = (dx / tangent_len, dy / tangent_len)
            segments.append(
                {
                    "s0": s0,
                    "s1": s1,
                    "x0": x0,
                    "y0": y0,
                    "dx": dx,
                    "dy": dy,
                    "span": span if span > 0.0 else tangent_len,
                    "length_sq": length_sq,
                    "tx": tangent[0],
                    "ty": tangent[1],
                }
            )

        if not segments:
            return

        self._segments = segments
        self._s_vals = s_raw
        self.is_valid = True

    def _segment_index_from_s(self, target_s: float) -> int:
        if not self._segments:
            return 0
        if target_s <= self._segments[0]["s0"]:
            return 0
        if target_s >= self._segments[-1]["s1"]:
            return len(self._segments) - 1
        idx = bisect.bisect_right(self._s_vals, target_s) - 1
        return max(0, min(idx, len(self._segments) - 1))

    def project(
        self,
        px: float,
        py: float,
        *,
        approx_s: Optional[float] = None,
        prev_index: Optional[int] = None,
    ) -> Optional[_ProjectionResult]:
        if not self.is_valid or not self._segments:
            return None

        if prev_index is not None:
            start_idx = max(0, min(prev_index, len(self._segments) - 1))
        elif approx_s is not None:
            start_idx = self._segment_index_from_s(approx_s)
        else:
            start_idx = 0

        best: Optional[_ProjectionResult] = None
        best_dist: float = float("inf")

        max_radius = max(len(self._segments), 1)
        radius = 1
        while radius <= max_radius:
            lo = max(0, start_idx - radius)
            hi = min(len(self._segments), start_idx + radius + 1)
            for idx in range(lo, hi):
                seg = self._segments[idx]
                if seg["length_sq"] <= 1e-12:
                    continue
                vx = seg["dx"]
                vy = seg["dy"]
                rel_x = px - seg["x0"]
                rel_y = py - seg["y0"]
                seg_len_sq = seg["length_sq"]
                t = (rel_x * vx + rel_y * vy) / seg_len_sq
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                proj_x = seg["x0"] + vx * t
                proj_y = seg["y0"] + vy * t
                dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
                s_val = seg["s0"] + seg["span"] * t
                if dist_sq < best_dist - 1e-12:
                    best_dist = dist_sq
                    best = _ProjectionResult(
                        s=s_val,
                        x=proj_x,
                        y=proj_y,
                        tangent=(seg["tx"], seg["ty"]),
                        segment_index=idx,
                    )
            if best is not None:
                break
            radius *= 2

        if best is not None:
            return best

        # Fallback: exhaustive search to avoid missing projections due to radius limits.
        for idx, seg in enumerate(self._segments):
            if seg["length_sq"] <= 1e-12:
                continue
            vx = seg["dx"]
            vy = seg["dy"]
            rel_x = px - seg["x0"]
            rel_y = py - seg["y0"]
            seg_len_sq = seg["length_sq"]
            t = (rel_x * vx + rel_y * vy) / seg_len_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            proj_x = seg["x0"] + vx * t
            proj_y = seg["y0"] + vy * t
            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            s_val = seg["s0"] + seg["span"] * t
            if dist_sq < best_dist - 1e-12:
                best_dist = dist_sq
                best = _ProjectionResult(
                    s=s_val,
                    x=proj_x,
                    y=proj_y,
                    tangent=(seg["tx"], seg["ty"]),
                    segment_index=idx,
                )

        return best

    def interpolate(
        self, target_s: float, prev_index: Optional[int] = None
    ) -> Optional[Tuple[float, float, Tuple[float, float], int]]:
        if not self.is_valid or not self._segments:
            return None

        idx = self._segment_index_from_s(target_s)
        if prev_index is not None:
            idx = max(0, min(prev_index, len(self._segments) - 1))
            seg = self._segments[idx]
            if target_s < seg["s0"] - 1e-6 and idx > 0:
                idx = self._segment_index_from_s(target_s)
            elif target_s > seg["s1"] + 1e-6 and idx < len(self._segments) - 1:
                idx = self._segment_index_from_s(target_s)

        seg = self._segments[idx]
        span = seg["span"]
        if span <= 1e-9:
            t = 0.0
        else:
            t = (target_s - seg["s0"]) / span
            t = min(1.0, max(0.0, t))

        x = seg["x0"] + seg["dx"] * t
        y = seg["y0"] + seg["dy"] * t
        tangent = (seg["tx"], seg["ty"])
        return x, y, tangent, idx


def _continuous_normal(
    tangent: Tuple[float, float], prev_normal: Optional[Tuple[float, float]]
) -> Tuple[float, float]:
    tx, ty = tangent
    nx = -ty
    ny = tx
    norm = math.hypot(nx, ny)
    if norm <= 1e-12:
        if prev_normal is not None:
            return prev_normal
        return (0.0, 1.0)
    nx /= norm
    ny /= norm
    if prev_normal is not None:
        dot = nx * prev_normal[0] + ny * prev_normal[1]
        if dot < 0.0:
            nx = -nx
            ny = -ny
    return (nx, ny)


def _reproject_sequence(
    points: List[Tuple[float, float, float, float, Optional[float], Optional[float]]],
    projector: _CenterlineProjector,
    *,
    monotonic_tolerance: float = 1e-4,
) -> List[List[Dict[str, float]]]:
    sequences: List[List[Dict[str, float]]] = []
    current: List[Dict[str, float]] = []
    prev_s: Optional[float] = None
    prev_index: Optional[int] = None
    prev_normal: Optional[Tuple[float, float]] = None

    for s_guess, px, py, pz, shape_idx, abs_offset in points:
        projection = projector.project(px, py, approx_s=s_guess, prev_index=prev_index)
        if projection is None:
            continue

        normal = _continuous_normal(projection.tangent, prev_normal)
        prev_normal = normal

        dx = px - projection.x
        dy = py - projection.y
        offset = dx * normal[0] + dy * normal[1]

        sample = {
            "s": projection.s,
            "offset": offset,
            "z": pz,
            "shape_index": shape_idx if shape_idx is not None else None,
            "absolute_offset": abs_offset if abs_offset is not None else None,
        }

        if prev_s is not None and sample["s"] < prev_s - monotonic_tolerance:
            if len(current) >= 2:
                sequences.append(current)
            current = []
            prev_normal = None
        current.append(sample)
        prev_s = sample["s"]
        prev_index = projection.segment_index

    if len(current) >= 2:
        sequences.append(current)

    return sequences


def _reconstruct_geometry_from_centerline(
    samples: List[Dict[str, float]],
    projector: _CenterlineProjector,
    *,
    resample_step: float = CENTERLINE_RESAMPLE_STEP,
    smooth_window: float = CENTERLINE_SMOOTHING_WINDOW,
) -> Optional[Dict[str, List[float]]]:
    if len(samples) < 2:
        return None

    ordered = sorted(samples, key=lambda item: item["s"])
    dedup: List[Dict[str, float]] = []
    for sample in ordered:
        if dedup and abs(sample["s"] - dedup[-1]["s"]) <= 1e-6:
            dedup[-1] = sample
        else:
            dedup.append(sample)

    if len(dedup) < 2:
        return None

    s_vals = [float(item["s"]) for item in dedup]
    offsets = [float(item["offset"]) for item in dedup]
    z_vals = [float(item["z"]) for item in dedup]
    shape_vals = [item.get("shape_index") for item in dedup]
    abs_offsets = [item.get("absolute_offset") for item in dedup]

    offsets_smoothed = _smooth_series(offsets, s_vals, smooth_window)
    z_smoothed = _smooth_series(z_vals, s_vals, smooth_window)

    start_s = s_vals[0]
    end_s = s_vals[-1]
    targets: List[float] = []
    current = start_s
    while current < end_s:
        targets.append(current)
        current += resample_step
    if not targets or abs(targets[-1] - end_s) > 1e-6:
        targets.append(end_s)

    combined = sorted(s_vals + targets)
    dedup_targets: List[float] = []
    for value in combined:
        if not dedup_targets or abs(value - dedup_targets[-1]) > 1e-9:
            dedup_targets.append(value)
    targets = dedup_targets

    offset_interp = _interp_values(s_vals, offsets_smoothed, targets)
    z_interp = _interp_values(s_vals, z_smoothed, targets)
    shape_interp = _interp_optional_values(s_vals, shape_vals, targets)
    abs_offset_interp = _interp_optional_values(s_vals, abs_offsets, targets)

    x_vals: List[float] = []
    y_vals: List[float] = []
    prev_normal: Optional[Tuple[float, float]] = None
    prev_index: Optional[int] = None

    for idx, target_s in enumerate(targets):
        interp = projector.interpolate(target_s, prev_index=prev_index)
        if interp is None:
            return None
        base_x, base_y, tangent, seg_index = interp
        normal = _continuous_normal(tangent, prev_normal)
        prev_normal = normal
        prev_index = seg_index
        offset = offset_interp[idx]
        x_vals.append(base_x + offset * normal[0])
        y_vals.append(base_y + offset * normal[1])

    geometry = {
        "s": targets,
        "x": x_vals,
        "y": y_vals,
        "z": z_interp,
    }

    curvature = _estimate_discrete_curvature(
        [(s, x, y, z, None, None) for s, x, y, z in zip(targets, x_vals, y_vals, z_interp)]
    )
    if curvature:
        geometry["curvature"] = curvature

    if any(val is not None for val in shape_interp):
        geometry["shape_index"] = shape_interp
    if any(val is not None for val in abs_offset_interp):
        geometry["absolute_offset"] = abs_offset_interp

    return geometry
MIN_POLY_DEGREE = 3
MAX_POLY_DEGREE = 5


def _polyval(coeffs: List[float], values: Iterable[float]) -> List[float]:
    result: List[float] = []
    for val in values:
        acc = 0.0
        for coeff in coeffs:
            acc = acc * val + coeff
        result.append(float(acc))
    return result


def _polyder(coeffs: List[float], order: int = 1) -> List[float]:
    deriv = list(coeffs)
    for _ in range(order):
        if len(deriv) <= 1:
            return [0.0]
        next_deriv: List[float] = []
        degree = len(deriv) - 1
        for idx, coeff in enumerate(deriv[:-1]):
            power = degree - idx
            next_deriv.append(coeff * power)
        deriv = next_deriv
    return deriv if deriv else [0.0]


def _solve_linear_system(matrix: List[List[float]], rhs: List[float]) -> Optional[List[float]]:
    size = len(rhs)
    augmented = [row[:] + [rhs[idx]] for idx, row in enumerate(matrix)]

    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(augmented[r][col]))
        pivot_val = augmented[pivot_row][col]
        if abs(pivot_val) <= 1e-12:
            return None
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot_val = augmented[col][col]
        for j in range(col, size + 1):
            augmented[col][j] /= pivot_val

        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) <= 1e-12:
                continue
            for j in range(col, size + 1):
                augmented[row][j] -= factor * augmented[col][j]

    return [augmented[i][size] for i in range(size)]


def _polyfit(ts: List[float], values: List[float], degree: int) -> Optional[List[float]]:
    if degree < 0:
        return None
    count = len(ts)
    cols = degree + 1
    vandermonde = [[t ** (degree - idx) for idx in range(cols)] for t in ts]

    normal: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(cols)]
    rhs: List[float] = [0.0 for _ in range(cols)]

    for row_idx in range(count):
        row = vandermonde[row_idx]
        value = values[row_idx]
        for i in range(cols):
            rhs[i] += row[i] * value
            for j in range(cols):
                normal[i][j] += row[i] * row[j]

    solution = _solve_linear_system(normal, rhs)
    return solution


def _interp_values(xs: List[float], ys: List[float], targets: List[float]) -> List[float]:
    if not xs:
        return [0.0 for _ in targets]
    result: List[float] = []
    last_idx = len(xs) - 1
    for val in targets:
        if val <= xs[0]:
            result.append(float(ys[0]))
            continue
        if val >= xs[last_idx]:
            result.append(float(ys[last_idx]))
            continue
        idx = bisect.bisect_left(xs, val) - 1
        if idx < 0:
            idx = 0
        next_idx = min(idx + 1, last_idx)
        x0 = xs[idx]
        x1 = xs[next_idx]
        y0 = ys[idx]
        y1 = ys[next_idx]
        if abs(x1 - x0) <= 1e-12:
            result.append(float(y0))
            continue
        ratio = (val - x0) / (x1 - x0)
        result.append(float(y0 + (y1 - y0) * ratio))
    return result


def _interp_optional_values(
    xs: List[float], values: List[Optional[float]], targets: List[float]
) -> List[Optional[float]]:
    defined = [
        (float(x), float(val))
        for x, val in zip(xs, values)
        if val is not None and math.isfinite(float(val))
    ]
    if not defined:
        return [None for _ in targets]
    defined.sort(key=lambda item: item[0])
    xs_def = [item[0] for item in defined]
    ys_def = [item[1] for item in defined]
    if len(xs_def) == 1:
        return [ys_def[0] for _ in targets]
    return [
        _interp_values(xs_def, ys_def, [target])[0]
        if target >= xs_def[0] and target <= xs_def[-1]
        else (
            ys_def[0]
            if target < xs_def[0]
            else ys_def[-1]
        )
        for target in targets
    ]


def _select_polynomial_degree(sample_count: int) -> Optional[int]:
    """Return an appropriate polynomial degree for ``sample_count`` points."""

    if sample_count < 2:
        return None

    max_supported = min(MAX_POLY_DEGREE, sample_count - 1)
    if max_supported < 1:
        return None
    if max_supported < MIN_POLY_DEGREE:
        return max_supported
    return max(MIN_POLY_DEGREE, min(MAX_POLY_DEGREE, max_supported))


def _resample_sequence_with_polynomial(
    points: List[Tuple[float, float, float, float, Optional[float], Optional[float]]],
    *,
    step: float = RESAMPLE_STEP_METERS,
) -> Optional[Dict[str, List[float] | List[Optional[float]]]]:
    """Fit a parametric polynomial to ``points`` and resample curvature."""

    if len(points) < 2:
        return None

    s_vals = [float(p[0]) for p in points]
    x_vals = [float(p[1]) for p in points]
    y_vals = [float(p[2]) for p in points]
    z_vals = [float(p[3]) for p in points]

    seg_lengths: List[float] = []
    total_length = 0.0
    for idx in range(1, len(points)):
        dx = x_vals[idx] - x_vals[idx - 1]
        dy = y_vals[idx] - y_vals[idx - 1]
        length = math.hypot(dx, dy)
        seg_lengths.append(length)
        total_length += length
    if not math.isfinite(total_length) or total_length <= 0.0:
        return None

    arc_lengths: List[float] = [0.0]
    for length in seg_lengths:
        arc_lengths.append(arc_lengths[-1] + length)
    degree = _select_polynomial_degree(len(points))
    if degree is None or degree < 1:
        return None

    t_vals = [arc / total_length for arc in arc_lengths]

    poly_x = _polyfit(t_vals, x_vals, degree)
    poly_y = _polyfit(t_vals, y_vals, degree)
    if poly_x is None or poly_y is None:
        return None

    dpoly_x = _polyder(poly_x, 1)
    ddpoly_x = _polyder(poly_x, 2)
    dpoly_y = _polyder(poly_y, 1)
    ddpoly_y = _polyder(poly_y, 2)

    targets: List[float] = []
    current = 0.0
    while current < total_length:
        targets.append(current)
        current += step
    if not targets or abs(targets[-1] - total_length) > 1e-6:
        targets.append(total_length)

    combined = sorted(targets + arc_lengths)
    dedup_targets: List[float] = []
    for value in combined:
        if not dedup_targets or abs(value - dedup_targets[-1]) > 1e-9:
            dedup_targets.append(value)
    targets = dedup_targets

    t_new = [target / total_length for target in targets]
    x_new = _polyval(poly_x, t_new)
    y_new = _polyval(poly_y, t_new)
    z_new = _interp_values(arc_lengths, z_vals, targets)
    s_new = _interp_values(arc_lengths, s_vals, targets)
    shape_vals = [p[4] if p[4] is not None else None for p in points]
    abs_offsets = [p[5] if p[5] is not None else None for p in points]
    shape_new = _interp_optional_values(arc_lengths, shape_vals, targets)
    offset_new = _interp_optional_values(arc_lengths, abs_offsets, targets)

    x_prime = _polyval(dpoly_x, t_new)
    y_prime = _polyval(dpoly_y, t_new)
    x_double = _polyval(ddpoly_x, t_new)
    y_double = _polyval(ddpoly_y, t_new)

    curvature: List[float] = []
    for xp, yp, xd, yd in zip(x_prime, y_prime, x_double, y_double):
        denom = (xp * xp + yp * yp) ** 1.5
        if denom <= 1e-9:
            curvature.append(0.0)
        else:
            curvature.append((xp * yd - yp * xd) / denom)

    return {
        "s": s_new,
        "x": x_new,
        "y": y_new,
        "z": z_new,
        "curvature": curvature,
        "shape_index": shape_new,
        "absolute_offset": offset_new,
    }


def _estimate_discrete_curvature(
    points: List[Tuple[float, float, float, float, Optional[float], Optional[float]]]
) -> List[float]:
    if not points:
        return []

    count = len(points)
    if count == 1:
        return [0.0]

    result: List[float] = []
    for idx in range(count):
        prev_idx = max(idx - 1, 0)
        next_idx = min(idx + 1, count - 1)
        if prev_idx == idx or next_idx == idx:
            result.append(0.0)
            continue

        _, x_prev, y_prev, _, _, _ = points[prev_idx]
        _, x_curr, y_curr, _, _, _ = points[idx]
        _, x_next, y_next, _, _, _ = points[next_idx]

        vec_prev = (x_curr - x_prev, y_curr - y_prev)
        vec_next = (x_next - x_curr, y_next - y_curr)
        chord = (x_next - x_prev, y_next - y_prev)

        len_prev = math.hypot(vec_prev[0], vec_prev[1])
        len_next = math.hypot(vec_next[0], vec_next[1])
        len_chord = math.hypot(chord[0], chord[1])

        denom = len_prev * len_next * len_chord
        if denom <= 1e-12:
            result.append(0.0)
            continue

        area_twice = vec_prev[0] * chord[1] - vec_prev[1] * chord[0]
        curvature = 2.0 * area_twice / denom
        result.append(float(curvature))

    return result

