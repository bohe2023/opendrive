"""CSVから区画線ジオメトリを解析するためのユーティリティ。"""

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
    except Exception:  # pragma: no cover - 防御的な分岐
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
    """正規化したラインIDからローカルXYポリラインへの写像を構築する。"""

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

    # かつて ``curvature_samples`` は生の曲率計測値を注入するために使われていた。
    # 現在は平滑化パイプラインがレーンジオメトリから直接曲率を求めるため、
    # 外部から渡されたサンプルは無視する。既存呼び出しとの互換性維持のため
    # 引数自体は残してある。

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
                except Exception:  # pragma: no cover - 防御的な分岐
                    raw_value = None
                raw_count = _to_float(raw_value)
                if raw_count is not None and raw_count > 0:
                    try:
                        count_val = int(round(raw_count))
                    except Exception:  # pragma: no cover - 防御的な分岐
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
        except Exception:  # pragma: no cover - 防御的な分岐
            continue

        if len(x_vals) != len(z_vals) or len(x_vals) != len(offsets_raw):
            continue

        if len(shape_indices) != len(offsets_raw):
            if len(shape_indices) < len(offsets_raw):
                shape_indices = shape_indices + [None] * (len(offsets_raw) - len(shape_indices))
            else:
                shape_indices = shape_indices[: len(offsets_raw)]

        # 日本のデータセットでは複数の車道区間で同じ ``line_id`` が再利用される。
        # オフセットが0へ戻るたびにジオメトリも先頭へ巻き戻り、ビューア上では
        # 麺のように絡んだ線になる。s方向の値が明確に逆行したタイミングで分割し、
        # 生成するポリラインが常に単調増加になるよう保つ。
        sequences: List[List[Tuple[float, float, float, float, Optional[float], Optional[float]]]] = []
        current: List[Tuple[float, float, float, float, Optional[float], Optional[float]]] = []
        last_s: Optional[float] = None
        reset_threshold = 1e-4  # 実際のリセットを見落とさずサブミリの揺らぎを許容

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
                except Exception:  # pragma: no cover - 防御的な分岐
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
                    # 分割後にゼロ長へ潰れてしまう重複サンプルはスキップする。
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
                        "curvature": [float(val) for val in resampled.get("curvature", [])],
                        "tangent_x": [float(val) for val in resampled.get("tangent_x", [])],
                        "tangent_y": [float(val) for val in resampled.get("tangent_y", [])],
                    }

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

                tangent_x = [float(val) for val in resampled.get("tangent_x", [])]
                tangent_y = [float(val) for val in resampled.get("tangent_y", [])]

                geom_entry = {
                    "s": resampled_s,
                    "x": resampled_x,
                    "y": resampled_y,
                    "z": resampled_z,
                    "curvature": resampled_curvature,
                    "tangent_x": tangent_x,
                    "tangent_y": tangent_y,
                }
                if not any(abs(val) > 1e-12 for val in resampled_curvature):
                    geom_entry["curvature"] = [0.0 for _ in resampled_curvature]
                geoms.append(geom_entry)
                continue

            signature = _geometry_signature([(p[0], p[1], p[2], p[3]) for p in points])
            if any(
                _geometry_signature(list(zip(g["s"], g["x"], g["y"], g["z"]))) == signature
                for g in geoms
            ):
                continue

            s_plain = [p[0] for p in points]
            x_plain = [p[1] for p in points]
            y_plain = [p[2] for p in points]
            z_plain = [p[3] for p in points]

            tangent_x, tangent_y, curvature_vals = _derive_path_kinematics(
                s_plain,
                x_plain,
                y_plain,
            )

            geom_entry = {
                "s": s_plain,
                "x": x_plain,
                "y": y_plain,
                "z": z_plain,
                "curvature": curvature_vals,
                "tangent_x": tangent_x,
                "tangent_y": tangent_y,
            }

            geoms.append(geom_entry)

    return lookup

RESAMPLE_STEP_METERS = 10.0
RESAMPLE_MIN_STEP_METERS = 0.5
RESAMPLE_ARC_ERROR_TOLERANCE = 0.02
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
    """センターライン上への弧長射影を提供する補助クラス。"""

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
        except Exception:  # pragma: no cover - 防御的な分岐
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
                # 退化区間の場合は可能であれば直前の接線を再利用する。
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

        # フォールバック: 半径制限で射影を見落とさないよう全探索する。
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

    tangent_x, tangent_y, curvature = _derive_path_kinematics(
        targets,
        x_vals,
        y_vals,
        smooth_window=smooth_window,
    )

    geometry = {
        "s": targets,
        "x": x_vals,
        "y": y_vals,
        "z": z_interp,
        "curvature": curvature,
        "tangent_x": tangent_x,
        "tangent_y": tangent_y,
    }

    if any(val is not None for val in shape_interp):
        geometry["shape_index"] = shape_interp
    if any(val is not None for val in abs_offset_interp):
        geometry["absolute_offset"] = abs_offset_interp

    return geometry
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


def _derive_path_kinematics(
    s_vals: List[float],
    x_vals: List[float],
    y_vals: List[float],
    *,
    reference_signs: Optional[List[Optional[float]]] = None,
    reference_values: Optional[List[Optional[float]]] = None,
    smooth_window: Optional[float] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """単調な弧長パラメータに対する単位接線と曲率を返す。"""

    count = len(s_vals)
    if count == 0:
        return [], [], []

    tangent_x: List[float] = []
    tangent_y: List[float] = []

    for idx in range(count):
        if idx == 0:
            idx_prev = idx
            idx_next = min(idx + 1, count - 1)
        elif idx == count - 1:
            idx_prev = max(idx - 1, 0)
            idx_next = idx
        else:
            idx_prev = idx - 1
            idx_next = idx + 1

        ds = s_vals[idx_next] - s_vals[idx_prev]
        if abs(ds) <= 1e-12:
            if idx_next != idx:
                ds = s_vals[idx_next] - s_vals[idx]
                dx = x_vals[idx_next] - x_vals[idx]
                dy = y_vals[idx_next] - y_vals[idx]
            elif idx_prev != idx:
                ds = s_vals[idx] - s_vals[idx_prev]
                dx = x_vals[idx] - x_vals[idx_prev]
                dy = y_vals[idx] - y_vals[idx_prev]
            else:
                ds = 1.0
                dx = 1.0
                dy = 0.0
        else:
            dx = x_vals[idx_next] - x_vals[idx_prev]
            dy = y_vals[idx_next] - y_vals[idx_prev]

        if abs(ds) <= 1e-12:
            vx, vy = 1.0, 0.0
        else:
            vx = dx / ds
            vy = dy / ds

        length = math.hypot(vx, vy)
        if length <= 1e-12:
            if tangent_x:
                vx, vy = tangent_x[-1], tangent_y[-1]
            else:
                vx, vy = 1.0, 0.0
        else:
            vx /= length
            vy /= length

        tangent_x.append(vx)
        tangent_y.append(vy)

    curvature = _estimate_discrete_curvature(
        [
            (s_vals[idx], x_vals[idx], y_vals[idx], 0.0, None, None)
            for idx in range(count)
        ]
    )

    if smooth_window is not None and smooth_window > 0.0 and len(curvature) >= 3:
        curvature = _smooth_series(curvature, s_vals, float(smooth_window))

    if reference_values is not None and len(reference_values) == len(curvature):
        for idx, ref in enumerate(reference_values):
            if ref is None:
                continue
            try:
                ref_val = float(ref)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ref_val) or abs(ref_val) <= 1e-12:
                continue
            curvature[idx] = ref_val

    if reference_signs is not None and len(reference_signs) == len(curvature):
        for idx, sign in enumerate(reference_signs):
            if sign is None or sign == 0.0:
                continue
            curv = curvature[idx]
            if curv * sign < 0.0:
                curvature[idx] = abs(curv) * math.copysign(1.0, sign)

    if len(curvature) > 1:
        if abs(curvature[0]) <= 1e-9:
            curvature[0] = curvature[1]
        if abs(curvature[-1]) <= 1e-9:
            curvature[-1] = curvature[-2]

    return tangent_x, tangent_y, curvature


def _resample_sequence_with_polynomial(
    points: List[Tuple[float, float, float, float, Optional[float], Optional[float]]],
    *,
    step: float = RESAMPLE_STEP_METERS,
) -> Optional[Dict[str, List[float] | List[Optional[float]]]]:
    """点列を三次ベジエで近似し曲率を再サンプリングする。"""

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

    # 各サンプル点における弧長に関する微分係数。端点は片側差分へフォールバックし、
    # 内点は隣接区間の平均勾配を用いて接線のC1連続性を確保する。
    dx_ds: List[float] = []
    dy_ds: List[float] = []
    last_idx = len(points) - 1
    for idx in range(len(points)):
        if idx == 0:
            span = max(arc_lengths[1] - arc_lengths[0], 1e-9)
            dx_ds.append((x_vals[1] - x_vals[0]) / span)
            dy_ds.append((y_vals[1] - y_vals[0]) / span)
            continue
        if idx == last_idx:
            span = max(arc_lengths[-1] - arc_lengths[-2], 1e-9)
            dx_ds.append((x_vals[-1] - x_vals[-2]) / span)
            dy_ds.append((y_vals[-1] - y_vals[-2]) / span)
            continue

        span_total = arc_lengths[idx + 1] - arc_lengths[idx - 1]
        if span_total <= 1e-9:
            span_prev = arc_lengths[idx] - arc_lengths[idx - 1]
            span_next = arc_lengths[idx + 1] - arc_lengths[idx]
            dx_prev = (x_vals[idx] - x_vals[idx - 1]) / max(span_prev, 1e-9)
            dx_next = (x_vals[idx + 1] - x_vals[idx]) / max(span_next, 1e-9)
            dy_prev = (y_vals[idx] - y_vals[idx - 1]) / max(span_prev, 1e-9)
            dy_next = (y_vals[idx + 1] - y_vals[idx]) / max(span_next, 1e-9)
            dx_ds.append(0.5 * (dx_prev + dx_next))
            dy_ds.append(0.5 * (dy_prev + dy_next))
        else:
            dx_ds.append((x_vals[idx + 1] - x_vals[idx - 1]) / span_total)
            dy_ds.append((y_vals[idx + 1] - y_vals[idx - 1]) / span_total)

    discrete_curv = _estimate_discrete_curvature(points)
    discrete_curv_for_sign = list(discrete_curv)
    if len(discrete_curv_for_sign) > 1:
        discrete_curv_for_sign[0] = discrete_curv_for_sign[1]
        discrete_curv_for_sign[-1] = discrete_curv_for_sign[-2]

    segments: List[Dict[str, Any]] = []
    for idx in range(len(points) - 1):
        length = arc_lengths[idx + 1] - arc_lengths[idx]

        segment_sign = 0.0
        if discrete_curv_for_sign:
            for candidate in (
                discrete_curv_for_sign[idx] if idx < len(discrete_curv_for_sign) else None,
                discrete_curv_for_sign[idx + 1]
                if idx + 1 < len(discrete_curv_for_sign)
                else None,
            ):
                if candidate is not None and abs(candidate) > 1e-12:
                    segment_sign = math.copysign(1.0, candidate)
                    break
        if segment_sign == 0.0 and idx + 2 < len(points):
            v1x = x_vals[idx + 1] - x_vals[idx]
            v1y = y_vals[idx + 1] - y_vals[idx]
            v2x = x_vals[idx + 2] - x_vals[idx + 1]
            v2y = y_vals[idx + 2] - y_vals[idx + 1]
            cross = v1x * v2y - v1y * v2x
            if abs(cross) > 1e-12:
                segment_sign = math.copysign(1.0, cross)
        if segment_sign == 0.0 and idx > 0:
            v0x = x_vals[idx] - x_vals[idx - 1]
            v0y = y_vals[idx] - y_vals[idx - 1]
            v1x = x_vals[idx + 1] - x_vals[idx]
            v1y = y_vals[idx + 1] - y_vals[idx]
            cross = v0x * v1y - v0y * v1x
            if abs(cross) > 1e-12:
                segment_sign = math.copysign(1.0, cross)

        if length <= 1e-9:
            continue
        x0 = x_vals[idx]
        y0 = y_vals[idx]
        x3 = x_vals[idx + 1]
        y3 = y_vals[idx + 1]
        control1 = (
            x0 + dx_ds[idx] * length / 3.0,
            y0 + dy_ds[idx] * length / 3.0,
        )
        control2 = (
            x3 - dx_ds[idx + 1] * length / 3.0,
            y3 - dy_ds[idx + 1] * length / 3.0,
        )
        segments.append(
            {
                "start": arc_lengths[idx],
                "end": arc_lengths[idx + 1],
                "p0": (x0, y0),
                "p1": control1,
                "p2": control2,
                "p3": (x3, y3),
                "sign": segment_sign,
            }
        )

    if not segments:
        return None

    curvature_magnitudes: List[float] = []
    for idx in range(len(points)):
        curv_val = 0.0
        if discrete_curv_for_sign and idx < len(discrete_curv_for_sign):
            candidate = discrete_curv_for_sign[idx]
            if candidate is not None:
                curv_val = abs(float(candidate))
        curvature_magnitudes.append(curv_val)

    targets: List[float] = []
    for idx, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        if not targets:
            targets.append(start)
        local_step = float(step)

        curvature_hint = 0.0
        for candidate in (
            curvature_magnitudes[idx] if idx < len(curvature_magnitudes) else None,
            curvature_magnitudes[idx + 1] if idx + 1 < len(curvature_magnitudes) else None,
        ):
            if candidate is None:
                continue
            curvature_hint = max(curvature_hint, float(candidate))

        if curvature_hint > 1e-9:
            max_span = math.sqrt(max(0.0, 8.0 * RESAMPLE_ARC_ERROR_TOLERANCE / curvature_hint))
            if max_span > 0.0:
                local_step = min(local_step, max_span)

        local_step = max(local_step, RESAMPLE_MIN_STEP_METERS)

        position = start + local_step
        while position < end - 1e-9:
            targets.append(position)
            position += local_step
        targets.append(end)

    dedup_targets: List[float] = []
    for value in targets:
        if not dedup_targets or abs(value - dedup_targets[-1]) > 1e-9:
            dedup_targets.append(value)
    targets = dedup_targets

    z_new = _interp_values(arc_lengths, z_vals, targets)
    s_new = _interp_values(arc_lengths, s_vals, targets)
    shape_vals = [p[4] if p[4] is not None else None for p in points]
    abs_offsets = [p[5] if p[5] is not None else None for p in points]
    shape_new = _interp_optional_values(arc_lengths, shape_vals, targets)
    offset_new = _interp_optional_values(arc_lengths, abs_offsets, targets)

    x_new: List[float] = []
    y_new: List[float] = []

    for idx_target, target in enumerate(targets):
        seg_idx = bisect.bisect_right(arc_lengths, target) - 1
        seg_idx = max(0, min(seg_idx, len(segments) - 1))
        seg = segments[seg_idx]
        length = max(seg["end"] - seg["start"], 1e-9)
        u = 0.0 if length <= 1e-9 else (target - seg["start"]) / length
        u = max(0.0, min(u, 1.0))

        px, py, dx_du, dy_du, ddx_duu, ddy_duu = _evaluate_bezier(seg, u)
        x_new.append(px)
        y_new.append(py)

        # 将来のチューニングに備えて微分評価を保持している。
        # 現状では再サンプルしたポリラインから曲率を算出している。
        _ = dx_du, dy_du, ddx_duu, ddy_duu

    resampled_points = [
        (s_val, x_val, y_val, z_val, None, None)
        for s_val, x_val, y_val, z_val in zip(s_new, x_new, y_new, z_new)
    ]
    reference_signs = _interp_optional_values(
        arc_lengths,
        [float(seg.get("sign", 0.0) or 0.0) for seg in segments],
        targets,
    )
    reference_curvature = _interp_values(
        s_vals,
        [
            float(val) if val is not None else 0.0
            for val in discrete_curv_for_sign
        ],
        s_new,
    ) if s_vals else [0.0 for _ in s_new]
    tangent_x, tangent_y, curvature = _derive_path_kinematics(
        s_new,
        x_new,
        y_new,
        reference_signs=reference_signs,
        reference_values=reference_curvature,
    )

    return {
        "s": s_new,
        "x": x_new,
        "y": y_new,
        "z": z_new,
        "curvature": curvature,
        "shape_index": shape_new,
        "absolute_offset": offset_new,
        "tangent_x": tangent_x,
        "tangent_y": tangent_y,
    }


def _evaluate_bezier(segment: Dict[str, Any], u: float) -> Tuple[float, float, float, float, float, float]:
    """三次ベジエ ``segment`` の位置と微係数を返す。"""

    p0x, p0y = segment["p0"]
    p1x, p1y = segment["p1"]
    p2x, p2y = segment["p2"]
    p3x, p3y = segment["p3"]

    om = 1.0 - u
    om2 = om * om
    u2 = u * u

    px = (
        om2 * om * p0x
        + 3.0 * om2 * u * p1x
        + 3.0 * om * u2 * p2x
        + u2 * u * p3x
    )
    py = (
        om2 * om * p0y
        + 3.0 * om2 * u * p1y
        + 3.0 * om * u2 * p2y
        + u2 * u * p3y
    )

    dx_du = 3.0 * (
        om2 * (p1x - p0x)
        + 2.0 * om * u * (p2x - p1x)
        + u2 * (p3x - p2x)
    )
    dy_du = 3.0 * (
        om2 * (p1y - p0y)
        + 2.0 * om * u * (p2y - p1y)
        + u2 * (p3y - p2y)
    )

    ddx_duu = 6.0 * (
        om * (p2x - 2.0 * p1x + p0x)
        + u * (p3x - 2.0 * p2x + p1x)
    )
    ddy_duu = 6.0 * (
        om * (p2y - 2.0 * p1y + p0y)
        + u * (p3y - 2.0 * p2y + p1y)
    )

    return px, py, dx_du, dy_du, ddx_duu, ddy_duu


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

