import bisect
import math
import statistics
import unicodedata
from typing import Callable, Iterable, List, Tuple, Optional, Any, Dict

from csv2xodr.simpletable import DataFrame
from csv2xodr.topology.core import _canonical_numeric


CURVATURE_RESAMPLE_STEP = 10.0
CURVATURE_MIN_POLY_DEGREE = 3
CURVATURE_MAX_POLY_DEGREE = 5


def _smooth_series(values: List[float], positions: List[float], window: float) -> List[float]:
    if len(values) <= 2 or len(values) != len(positions) or window <= 0.0:
        return list(values)

    half_window = max(window * 0.5, 1e-6)
    smoothed: List[float] = [0.0 for _ in values]

    left = 0
    right = 0
    running_sum = 0.0
    running_count = 0

    for idx, center in enumerate(positions):
        while left < len(values) and positions[left] < center - half_window:
            running_sum -= values[left]
            running_count -= 1
            left += 1
        while right < len(values) and positions[right] <= center + half_window:
            running_sum += values[right]
            running_count += 1
            right += 1
        if running_count > 0:
            smoothed[idx] = running_sum / running_count
        else:  # pragma: no cover - defensive fallback
            smoothed[idx] = values[idx]

    return smoothed

def _col_like(df: DataFrame, keyword: str):
    cols = [c for c in df.columns if keyword in c]
    return cols[0] if cols else None


def _find_column(df: DataFrame, *keywords: str, exclude: Tuple[str, ...] = ()) -> Optional[str]:
    lowered = [kw.lower() for kw in keywords]
    excluded = [kw.lower() for kw in exclude]
    for col in df.columns:
        stripped = col.strip()
        value = stripped.lower()
        if any(block in value for block in excluded):
            continue
        if all(keyword in value for keyword in lowered):
            return col
    return None


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion that accepts common locale specific formats."""

    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    # Normalise full-width characters that occasionally show up in JPN CSVs.
    normalised = unicodedata.normalize("NFKC", text)

    # Remove whitespace-like grouping characters before handling decimal/grouping
    # separators.  This covers plain spaces as well as non-breaking/thin spaces
    # that are often used as thousands separators in exported spreadsheets.
    grouping_chars = {" ", "\u00a0", "\u2009", "_"}
    compact = "".join(ch for ch in normalised if ch not in grouping_chars)

    # Detect whether commas should be treated as decimal or grouping
    # separators.  ``1,234.5`` → ``1234.5`` whereas ``12,5`` → ``12.5``.
    candidate = compact
    if "," in compact:
        if "." in compact or compact.count(",") > 1:
            candidate = compact.replace(",", "")
        else:
            head, tail = compact.split(",", 1)
            tail_digits = tail.isdigit()
            if tail_digits and tail and len(tail) % 3 == 0 and len(head) > 0:
                candidate = head + tail
            elif tail_digits:
                candidate = f"{head}.{tail}"
            else:
                candidate = compact.replace(",", "")

    try:
        return float(candidate)
    except ValueError:
        # As a last resort drop all commas.  This allows inputs such as
        # ``1,234`` to still parse even when the decimal/comma heuristic above
        # could not disambiguate the intent.
        fallback = compact.replace(",", "")
        if fallback != candidate:
            try:
                return float(fallback)
            except ValueError:
                return None
        return None
    except TypeError:  # pragma: no cover - defensive
        return None


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _advance_pose(
    start_x: float,
    start_y: float,
    start_hdg: float,
    curvature: float,
    length: float,
) -> Tuple[float, float, float]:
    """Integrate a constant-curvature segment from the provided pose."""

    if abs(curvature) <= 1e-12:
        end_x = start_x + length * math.cos(start_hdg)
        end_y = start_y + length * math.sin(start_hdg)
        end_hdg = start_hdg
    else:
        end_hdg = _normalize_angle(start_hdg + curvature * length)
        radius = 1.0 / curvature
        dx = radius * (math.sin(end_hdg) - math.sin(start_hdg))
        dy = -radius * (math.cos(end_hdg) - math.cos(start_hdg))
        end_x = start_x + dx
        end_y = start_y + dy
    return end_x, end_y, end_hdg


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


def _build_cubic_spline(knots: List[float], values: List[float]) -> Optional[Dict[str, List[float]]]:
    count = len(knots)
    if count < 2 or len(values) != count:
        return None

    spacing: List[float] = []
    for idx in range(count - 1):
        step = knots[idx + 1] - knots[idx]
        if step <= 0.0:
            return None
        spacing.append(step)

    alpha: List[float] = [0.0] * count
    for idx in range(1, count - 1):
        alpha[idx] = (
            3.0 * (values[idx + 1] - values[idx]) / spacing[idx]
            - 3.0 * (values[idx] - values[idx - 1]) / spacing[idx - 1]
        )

    lower: List[float] = [1.0] + [0.0] * (count - 1)
    upper: List[float] = [0.0] * count
    middle: List[float] = [0.0] * count

    for idx in range(1, count - 1):
        lower[idx] = (
            2.0 * (knots[idx + 1] - knots[idx - 1]) - spacing[idx - 1] * upper[idx - 1]
        )
        if abs(lower[idx]) <= 1e-12:
            return None
        upper[idx] = spacing[idx] / lower[idx]
        middle[idx] = (alpha[idx] - spacing[idx - 1] * middle[idx - 1]) / lower[idx]

    lower[-1] = 1.0
    middle[-1] = 0.0

    second: List[float] = [0.0] * count
    for idx in range(count - 2, -1, -1):
        second[idx] = middle[idx] - upper[idx] * second[idx + 1]

    coeff_a: List[float] = list(values[:-1])
    coeff_b: List[float] = [0.0] * (count - 1)
    coeff_c: List[float] = [0.0] * (count - 1)
    coeff_d: List[float] = [0.0] * (count - 1)

    for idx in range(count - 1):
        coeff_c[idx] = second[idx] * 0.5
        coeff_d[idx] = (second[idx + 1] - second[idx]) / (6.0 * spacing[idx])
        coeff_b[idx] = (
            (values[idx + 1] - values[idx]) / spacing[idx]
            - spacing[idx] * (second[idx + 1] + 2.0 * second[idx]) / 6.0
        )

    return {
        "knots": list(knots),
        "a": coeff_a,
        "b": coeff_b,
        "c": coeff_c,
        "d": coeff_d,
    }


def _eval_cubic_spline(
    spline: Dict[str, List[float]], value: float
) -> Tuple[float, float, float]:
    knots = spline["knots"]
    if value <= knots[0]:
        idx = 0
    elif value >= knots[-1]:
        idx = len(knots) - 2
    else:
        idx = max(0, bisect.bisect_right(knots, value) - 1)

    dt = value - knots[idx]
    a = spline["a"][idx]
    b = spline["b"][idx]
    c = spline["c"][idx]
    d = spline["d"][idx]

    position = ((d * dt + c) * dt + b) * dt + a
    first = (3.0 * d * dt + 2.0 * c) * dt + b
    second = 6.0 * d * dt + 2.0 * c
    return position, first, second


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
    ordered = sorted(zip(xs, ys), key=lambda item: item[0])
    xs_sorted = [float(item[0]) for item in ordered]
    ys_sorted = [float(item[1]) for item in ordered]
    result: List[float] = []
    last_idx = len(xs_sorted) - 1
    for val in targets:
        if val <= xs_sorted[0]:
            result.append(float(ys_sorted[0]))
            continue
        if val >= xs_sorted[last_idx]:
            result.append(float(ys_sorted[last_idx]))
            continue
        idx = bisect.bisect_left(xs_sorted, val) - 1
        if idx < 0:
            idx = 0
        next_idx = min(idx + 1, last_idx)
        x0 = xs_sorted[idx]
        x1 = xs_sorted[next_idx]
        y0 = ys_sorted[idx]
        y1 = ys_sorted[next_idx]
        if abs(x1 - x0) <= 1e-12:
            result.append(float(y0))
            continue
        ratio = (val - x0) / (x1 - x0)
        result.append(float(y0 + (y1 - y0) * ratio))
    return result


def _select_polynomial_degree(sample_count: int) -> Optional[int]:
    if sample_count < 2:
        return None
    max_supported = min(CURVATURE_MAX_POLY_DEGREE, sample_count - 1)
    if max_supported < 1:
        return None
    if max_supported < CURVATURE_MIN_POLY_DEGREE:
        return max_supported
    return max(CURVATURE_MIN_POLY_DEGREE, min(CURVATURE_MAX_POLY_DEGREE, max_supported))


def _resample_parametric_curve(
    s_vals: List[float],
    x_vals: List[float],
    y_vals: List[float],
    *,
    step: float = CURVATURE_RESAMPLE_STEP,
    preserve_targets: Optional[Iterable[float]] = None,
) -> Optional[Dict[str, List[float]]]:
    if len(s_vals) < 2 or len(x_vals) < 2 or len(y_vals) < 2:
        return None

    combined = sorted(zip(s_vals, x_vals, y_vals), key=lambda item: item[0])
    s_vals = [float(item[0]) for item in combined]
    x_vals = [float(item[1]) for item in combined]
    y_vals = [float(item[2]) for item in combined]

    seg_lengths: List[float] = []
    total_length = 0.0
    for idx in range(1, len(x_vals)):
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

    # 原始白线点位存在测量噪声时，直接拟合多项式会放大误差导致曲率
    # 高频震荡。先对坐标序列做弧长域上的平滑与等距重采样，可以抑制
    # 浮噪并让后续的导数运算更加稳定。
    if len(arc_lengths) == len(x_vals) == len(y_vals) and len(x_vals) >= 3:
        original_arc = list(arc_lengths)
        original_s = list(s_vals)
        smooth_window = max(step * 0.5, total_length * 0.05)
        smooth_window = min(smooth_window, total_length)
        if smooth_window > 0.0:
            x_vals = _smooth_series(x_vals, arc_lengths, smooth_window)
            y_vals = _smooth_series(y_vals, arc_lengths, smooth_window)

        # 通过沿弧长的线性插值生成等距采样点，避免原始测量的稠密/稀疏
        # 区域对多项式拟合的权重造成偏差。
        target_spacing = max(step * 0.25, total_length / max(len(arc_lengths) * 4, 1))
        target_spacing = min(target_spacing, max(total_length, 1e-6))
        if target_spacing > 0.0 and target_spacing < total_length:
            uniform_targets: List[float] = [0.0]
            while uniform_targets[-1] + target_spacing < total_length:
                uniform_targets.append(uniform_targets[-1] + target_spacing)
            if abs(uniform_targets[-1] - total_length) > 1e-6:
                uniform_targets.append(total_length)

            # 在原始序列上插值，确保重采样后的点仍然沿原曲线分布。
            x_vals = _interp_values(original_arc, x_vals, uniform_targets)
            y_vals = _interp_values(original_arc, y_vals, uniform_targets)
            s_vals = _interp_values(original_arc, original_s, uniform_targets)
            arc_lengths = uniform_targets

        # Savitzky–Golay 滤波可以进一步抑制高频噪声，同时保留真实曲率
        # 变化趋势；SciPy 不一定可用，因此在运行时按需导入。
        try:  # pragma: no cover - 依赖环境可能缺失
            from scipy.signal import savgol_filter  # type: ignore
        except Exception:  # pragma: no cover - 当 SciPy 不可用时静默跳过
            savgol_filter = None  # type: ignore

        if "savgol_filter" in locals() and callable(savgol_filter):  # type: ignore[name-defined]
            max_window = len(x_vals) if len(x_vals) % 2 == 1 else len(x_vals) - 1
            if max_window >= 5:
                approx_points_per_step = max(1, int(round(step / max(target_spacing, 1e-6))))
                window_length = min(max_window, approx_points_per_step * 2 + 1)
                if window_length < 5:
                    window_length = 5 if max_window >= 5 else max_window
                if window_length % 2 == 0:
                    window_length = max(5, window_length - 1)
                poly_order = min(3, max(1, window_length - 2))
                if window_length >= poly_order + 2 and window_length <= len(x_vals):
                    try:  # pragma: no cover - SciPy 调用
                        x_vals = savgol_filter(x_vals, window_length, poly_order).tolist()
                        y_vals = savgol_filter(y_vals, window_length, poly_order).tolist()
                    except Exception:  # pragma: no cover - 失败时回退原序列
                        pass

    filtered: List[Tuple[float, float, float, float]] = []
    last_arc: Optional[float] = None
    for arc_val, x_val, y_val, s_val in zip(arc_lengths, x_vals, y_vals, s_vals):
        if not (
            math.isfinite(arc_val)
            and math.isfinite(x_val)
            and math.isfinite(y_val)
            and math.isfinite(s_val)
        ):
            continue
        if last_arc is not None and abs(arc_val - last_arc) <= 1e-9:
            continue
        filtered.append((arc_val, x_val, y_val, s_val))
        last_arc = arc_val

    if len(filtered) < 2:
        return None

    arc_lengths = [item[0] for item in filtered]
    x_vals = [item[1] for item in filtered]
    y_vals = [item[2] for item in filtered]
    s_vals = [item[3] for item in filtered]

    origin = arc_lengths[0]
    if abs(origin) > 1e-9:
        arc_lengths = [arc - origin for arc in arc_lengths]
        if preserve_targets is not None:
            adjusted_targets: List[float] = []
            for value in preserve_targets:
                if value is None:
                    continue
                shifted = float(value) - origin
                if math.isfinite(shifted):
                    adjusted_targets.append(shifted)
            preserve_targets = adjusted_targets
    total_length = arc_lengths[-1]
    if total_length <= 0.0:
        return None

    spline_x = _build_cubic_spline(arc_lengths, x_vals)
    spline_y = _build_cubic_spline(arc_lengths, y_vals)

    if spline_x is None or spline_y is None:
        degree = _select_polynomial_degree(len(x_vals))
        if degree is None or degree < 1 or total_length <= 0.0:
            return None
        t_vals = [arc / total_length for arc in arc_lengths]
        poly_x = _polyfit(t_vals, x_vals, degree)
        poly_y = _polyfit(t_vals, y_vals, degree)
        if poly_x is None or poly_y is None:
            return None
        spline_targets: List[float] = []
        current = 0.0
        while current < total_length:
            spline_targets.append(current)
            current += step
        if not spline_targets or abs(spline_targets[-1] - total_length) > 1e-6:
            spline_targets.append(total_length)
        if preserve_targets is not None:
            for value in preserve_targets:
                if math.isfinite(value) and 0.0 <= value <= total_length:
                    spline_targets.append(float(value))
        dedup_targets: List[float] = []
        for value in sorted(spline_targets):
            if not dedup_targets or abs(value - dedup_targets[-1]) > 1e-9:
                dedup_targets.append(value)
        spline_targets = dedup_targets
        t_new = [target / total_length for target in spline_targets]
        x_new = _polyval(poly_x, t_new)
        y_new = _polyval(poly_y, t_new)
        dpoly_x = _polyder(poly_x, 1)
        dpoly_y = _polyder(poly_y, 1)
        ddpoly_x = _polyder(poly_x, 2)
        ddpoly_y = _polyder(poly_y, 2)
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
        if curvature:
            curvature = _smooth_series(curvature, spline_targets, step)
        s_interp = _interp_values(arc_lengths, s_vals, spline_targets)
        return {
            "s": s_interp,
            "curvature": curvature,
            "x": x_new,
            "y": y_new,
            "arc": spline_targets,
        }

    targets: List[float] = []
    current = 0.0
    while current < total_length:
        targets.append(current)
        current += step
    if not targets or abs(targets[-1] - total_length) > 1e-6:
        targets.append(total_length)

    if preserve_targets is not None:
        for value in preserve_targets:
            if math.isfinite(value) and 0.0 <= value <= total_length:
                targets.append(float(value))

    dedup_targets: List[float] = []
    for value in sorted(targets):
        if not dedup_targets or abs(value - dedup_targets[-1]) > 1e-9:
            dedup_targets.append(value)
    targets = dedup_targets

    x_new: List[float] = []
    y_new: List[float] = []
    x_prime: List[float] = []
    y_prime: List[float] = []
    x_double: List[float] = []
    y_double: List[float] = []
    for target in targets:
        x_pos, x_first, x_second = _eval_cubic_spline(spline_x, target)
        y_pos, y_first, y_second = _eval_cubic_spline(spline_y, target)
        x_new.append(x_pos)
        y_new.append(y_pos)
        x_prime.append(x_first)
        y_prime.append(y_first)
        x_double.append(x_second)
        y_double.append(y_second)

    curvature: List[float] = []
    for xp, yp, xd, yd in zip(x_prime, y_prime, x_double, y_double):
        denom = (xp * xp + yp * yp) ** 1.5
        if denom <= 1e-9:
            curvature.append(0.0)
        else:
            curvature.append((xp * yd - yp * xd) / denom)

    if curvature:
        curvature = _smooth_series(curvature, targets, step)

    s_interp = _interp_values(arc_lengths, s_vals, targets)

    return {
        "s": s_interp,
        "curvature": curvature,
        "x": x_new,
        "y": y_new,
        "arc": targets,
    }


def _find_height_column(df: DataFrame) -> Optional[str]:
    for col in df.columns:
        stripped = col.strip()
        lowered = stripped.lower()
        if "高さ" in stripped or "標高" in stripped or "height" in lowered or "[m]" in stripped:
            return col
    return None

def latlon_to_local_xy(lat: Iterable[float], lon: Iterable[float], lat0: float, lon0: float) -> Tuple[List[float], List[float]]:
    """
    Simple equirectangular projection to local XY [m].
    lat/lon can be numpy arrays in degrees.
    """
    R = 6378137.0
    lat_rad = [math.radians(v) for v in lat]
    lon_rad = [math.radians(v) for v in lon]
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x_vals = [
        (lon_v - lon0_rad) * math.cos((lat_v + lat0_rad) / 2.0) * R
        for lat_v, lon_v in zip(lat_rad, lon_rad)
    ]
    y_vals = [(lat_v - lat0_rad) * R for lat_v in lat_rad]
    return x_vals, y_vals


def select_best_path_id(df_line_geo: Optional[DataFrame]) -> Optional[str]:
    """Return the Path Id that best represents the reference alignment.

    The CSV datasets may contain multiple ``Path Id`` polylines (for example,
    one per carriageway).  The conversion pipeline expects to work on a single
    alignment, therefore we pick the longest available polyline and reuse that
    identifier for every other table.
    """

    if df_line_geo is None or len(df_line_geo) == 0:
        return None

    path_col = _col_like(df_line_geo, "Path")
    if path_col is None:
        return None

    unique_paths = df_line_geo[path_col].dropna().unique()
    if len(unique_paths) == 0:
        return None
    if len(unique_paths) == 1:
        raw_value = unique_paths[0]
        canonical = _canonical_numeric(raw_value, allow_negative=True)
        return canonical or (str(raw_value).strip() or None)

    lat_candidates = list(df_line_geo.filter(like="緯度").columns)
    if not lat_candidates:
        lat_candidates = list(df_line_geo.filter(like="Latitude").columns)
    lon_candidates = list(df_line_geo.filter(like="経度").columns)
    if not lon_candidates:
        lon_candidates = list(df_line_geo.filter(like="Longitude").columns)
    if not lat_candidates or not lon_candidates:
        return None

    lat_col = lat_candidates[0]
    lon_col = lon_candidates[0]

    path_points: Dict[str, List[Tuple[float, float]]] = {}
    order: List[str] = []

    for idx in range(len(df_line_geo)):
        row = df_line_geo.iloc[idx]
        path_raw = row[path_col]
        canonical = _canonical_numeric(path_raw, allow_negative=True)
        if canonical is None:
            canonical = str(path_raw).strip()
        if not canonical:
            continue

        lat_val = _to_float(row[lat_col])
        lon_val = _to_float(row[lon_col])
        if lat_val is None or lon_val is None:
            continue

        if canonical not in path_points:
            path_points[canonical] = []
            order.append(canonical)
        path_points[canonical].append((lat_val, lon_val))

    best_path: Optional[str] = None
    best_length = -1.0

    for path_id in order:
        points = path_points.get(path_id, [])
        if len(points) < 2:
            continue
        lat_vals = [lat for lat, _ in points]
        lon_vals = [lon for _, lon in points]
        lat0 = lat_vals[0]
        lon0 = lon_vals[0]
        x_vals, y_vals = latlon_to_local_xy(lat_vals, lon_vals, lat0, lon0)
        length = 0.0
        for i in range(len(x_vals) - 1):
            dx = x_vals[i + 1] - x_vals[i]
            dy = y_vals[i + 1] - y_vals[i]
            length += math.hypot(dx, dy)

        if length > best_length:
            best_length = length
            best_path = path_id

    return best_path


def filter_dataframe_by_path(df: Optional[DataFrame], path_id: Optional[str]) -> Optional[DataFrame]:
    """Return a DataFrame that only contains rows matching ``path_id``."""

    if df is None or len(df) == 0 or path_id is None:
        return df

    target_canonical = _canonical_numeric(path_id, allow_negative=True)
    target_text = str(path_id).strip()

    path_col = _col_like(df, "Path")
    if path_col is None:
        return df

    keep_mask: List[bool] = []
    series = df[path_col]
    for idx in range(len(df)):
        raw_value = series.iloc[idx]
        raw_canonical = _canonical_numeric(raw_value, allow_negative=True)
        match = False
        if raw_canonical is not None and target_canonical is not None:
            match = raw_canonical == target_canonical
        else:
            match = str(raw_value).strip() == target_text
        keep_mask.append(match)

    if not any(keep_mask):
        return df
    if all(keep_mask):
        return df

    filtered = df.loc[keep_mask]
    return filtered.reset_index(drop=True)

def build_centerline(df_line_geo: DataFrame, df_base: DataFrame):
    """
    Build centerline planView from PROFILETYPE_MPU_LINE_GEOMETRY (lat/lon series).
    Returns: centerline DataFrame [s,x,y,hdg], and (lat0, lon0)
    """
    if df_line_geo is None or len(df_line_geo) == 0:
        raise ValueError("line_geometry CSV is required")

    best_path_id = select_best_path_id(df_line_geo)
    if best_path_id is not None:
        filtered = filter_dataframe_by_path(df_line_geo, best_path_id)
        if filtered is not None:
            df_line_geo = filtered

    lat_col = df_line_geo.filter(like="緯度").columns[0]
    lon_col = df_line_geo.filter(like="経度").columns[0]

    # 部分数据集中同一 Offset 会重复多次，其内的多条曲线往往对应不同的
    # 车道边界。观察发现中心线可以通过 "3D 地物属性オプションフラグ" 标记
    # 出来，并且每个 Offset 分段的首批采样点数量等于 "形状要素点数"。为了
    # 保留沿纵向的几何细节，需要优先挑选中心线对应的记录，并在分段内部
    # 仅保留首批采样点，避免在不同车道间往返。
    offset_col = _col_like(df_line_geo, "Offset")
    end_offset_col = _find_column(df_line_geo, "end", "offset")
    flag_col: Optional[str] = None
    for column in df_line_geo.columns:
        lowered = column.lower()
        if "3d" in lowered and "オプション" in column:
            flag_col = column
            break

    shape_count_col = (
        _find_column(df_line_geo, "形状", "要素")
        or _find_column(df_line_geo, "shape", "count")
        or _find_column(df_line_geo, "shape", "points")
    )

    primary_rows: Optional[List[Dict[str, Any]]] = None
    if flag_col is not None and offset_col is not None:
        # ``DataFrame`` 是一个轻量包装，允许直接访问底层行列表。
        rows = getattr(df_line_geo, "_rows", None)
        if isinstance(rows, list):
            flag_values = []
            for row in rows:
                value = row.get(flag_col)
                if value is None:
                    continue
                text = str(value).strip()
                if text == "":
                    continue
                flag_values.append(text)

            chosen_flag: Optional[str] = None
            if flag_values:
                unique_flags = set(flag_values)
                # 数据集中中心线通常使用 4 标记，其次退化为 1/2/0 等其他值。
                preferred = ["4", "3", "2", "1", "0"]
                for candidate in preferred:
                    if candidate in unique_flags:
                        chosen_flag = candidate
                        break
                if chosen_flag is None:
                    # 尝试从数值最小的标记中挑选，避免完全失效。
                    numeric: List[Tuple[float, str]] = []
                    for flag in unique_flags:
                        try:
                            numeric.append((abs(float(flag)), flag))
                        except Exception:  # pragma: no cover - defensive
                            continue
                    if numeric:
                        numeric.sort(key=lambda item: item[0])
                        chosen_flag = numeric[0][1]

            if chosen_flag is not None:
                filtered_rows = [
                    row for row in rows if str(row.get(flag_col)).strip() == chosen_flag
                ]
                if filtered_rows:
                    # 记录 Offset 首次出现的顺序，随后在每个分段内仅保留前
                    # ``shape_count`` 个采样点，以复原沿参考线的细分信息。
                    order: List[str] = []
                    grouped: Dict[str, List[Dict[str, Any]]] = {}
                    for row in filtered_rows:
                        key = str(row.get(offset_col))
                        if key not in grouped:
                            order.append(key)
                            grouped[key] = []
                        grouped[key].append(row)

                    selected: List[Dict[str, Any]] = []
                    for key in order:
                        group = grouped.get(key, [])
                        if not group:
                            continue
                        limit: Optional[int] = None
                        if shape_count_col is not None:
                            raw_count = group[0].get(shape_count_col)
                            try:
                                limit = int(float(raw_count))
                            except Exception:  # pragma: no cover - defensive
                                limit = None
                        if limit is None or limit <= 0 or limit > len(group):
                            limit = len(group)

                        start_val = _to_float(group[0].get(offset_col))
                        end_val = (
                            _to_float(group[0].get(end_offset_col))
                            if end_offset_col is not None
                            else None
                        )
                        step = None
                        if (
                            start_val is not None
                            and end_val is not None
                            and limit > 1
                        ):
                            step = (end_val - start_val) / (limit - 1)

                        for idx in range(limit):
                            source = group[idx]
                            row_copy = dict(source)
                            if step is not None and offset_col is not None:
                                row_copy[offset_col] = start_val + step * idx
                            selected.append(row_copy)

                    if selected:
                        primary_rows = selected

    used_primary_rows = primary_rows is not None
    if used_primary_rows:
        df_line_geo = DataFrame(primary_rows, columns=df_line_geo.columns)
    if offset_col is not None:
        offsets_series = df_line_geo[offset_col].astype(float)
    else:
        offsets_series = None

    # Some providers emit one row per lane boundary for the same longitudinal
    # offset.  Averaging the duplicated offsets keeps the geometry focused on a
    # single centerline instead of weaving across multiple boundaries (which
    # renders as a cross/diamond artifact in the exported OpenDRIVE).
    if (
        offsets_series is not None
        and offsets_series.duplicated().any()
        and not used_primary_rows
    ):
        grouped = df_line_geo.groupby(offset_col, sort=True)[[lat_col, lon_col]].mean().reset_index()
        df_line_geo = grouped
        lat = [float(v) for v in grouped[lat_col].to_list()]
        lon = [float(v) for v in grouped[lon_col].to_list()]
        offsets = [float(v) for v in grouped[offset_col].astype(float).to_list()]
    else:
        lat = [float(v) for v in df_line_geo[lat_col].astype(float).to_list()]
        lon = [float(v) for v in df_line_geo[lon_col].astype(float).to_list()]
        offsets = offsets_series.to_list() if offsets_series is not None else None

    if offsets is not None and len(offsets) == len(lat):
        order = sorted(range(len(offsets)), key=lambda idx: offsets[idx])
        if any(idx != order[idx] for idx in range(len(order))):
            lat = [lat[idx] for idx in order]
            lon = [lon[idx] for idx in order]
            offsets = [offsets[idx] for idx in order]

    # choose origin
    if df_base is not None and len(df_base) > 0:
        lat0 = float(df_base.filter(like="緯度").iloc[0, 0])
        lon0 = float(df_base.filter(like="経度").iloc[0, 0])
    else:
        lat0 = sum(lat) / len(lat)
        lon0 = sum(lon) / len(lon)

    x, y = latlon_to_local_xy(lat, lon, lat0, lon0)

    # cumulative s & heading
    s = [0.0 for _ in x]
    hdg = [0.0 for _ in x]
    for i in range(1, len(x)):
        dx, dy = (x[i] - x[i - 1]), (y[i] - y[i - 1])
        ds = math.hypot(dx, dy)
        s[i] = s[i - 1] + ds
        hdg[i - 1] = math.atan2(dy, dx)
    if len(hdg) > 1:
        hdg[-1] = hdg[-2]

    offsets_column = None
    if offsets is not None and len(offsets) == len(s):
        offsets_f = [float(v) for v in offsets]
        if offsets_f:
            start = offsets_f[0]
            offsets_norm = [v - start for v in offsets_f]
            if s[-1] > 0:
                ratios = []
                for i in range(1, len(offsets_norm)):
                    delta_offset = offsets_norm[i] - offsets_norm[i - 1]
                    delta_s = s[i] - s[i - 1]
                    if delta_offset <= 0 or delta_s <= 0:
                        continue
                    ratios.append(delta_offset / delta_s)
                scale = 1.0
                if ratios:
                    typical = statistics.median(ratios)
                    if typical > 10.0:
                        scale = 0.01
                    elif typical < 0.1:
                        scale = 100.0
                if scale != 1.0:
                    offsets_norm = [v * scale for v in offsets_norm]
            offsets_column = offsets_norm

    if s:
        has_duplicates = False
        if len(s) > 1:
            prev = float(s[0])
            for idx in range(1, len(s)):
                curr = float(s[idx])
                if abs(curr - prev) <= 1e-9:
                    has_duplicates = True
                    break
                prev = curr

        if has_duplicates:
            cleaned: List[Tuple[float, float, float, float, Optional[float]]] = []
            for idx in range(len(s)):
                s_val = float(s[idx])
                x_val = float(x[idx])
                y_val = float(y[idx])
                hdg_val = float(hdg[idx])
                offset_val = (
                    float(offsets_column[idx]) if offsets_column is not None else None
                )

                if cleaned and abs(s_val - cleaned[-1][0]) <= 1e-9:
                    cleaned[-1] = (s_val, x_val, y_val, hdg_val, offset_val)
                else:
                    cleaned.append((s_val, x_val, y_val, hdg_val, offset_val))

            if cleaned:
                s = [item[0] for item in cleaned]
                x = [item[1] for item in cleaned]
                y = [item[2] for item in cleaned]
                hdg = [item[3] for item in cleaned]
                if offsets_column is not None:
                    offsets_column = [item[4] for item in cleaned]

    data = {"s": s, "x": x, "y": y, "hdg": hdg}
    if offsets_column is not None:
        data["s_offset"] = offsets_column

    center = DataFrame(data)
    return center, (lat0, lon0)


def build_offset_mapper(centerline: DataFrame) -> Callable[[float], float]:
    """Return a callable that maps CSV offsets to centreline arc-length."""

    if "s_offset" not in centerline.columns:
        return lambda value: float(value)

    offsets = [float(v) for v in centerline["s_offset"].to_list()]
    s_vals = [float(v) for v in centerline["s"].to_list()]

    if not offsets or len(offsets) != len(s_vals):
        return lambda value: float(value)

    first_slope: Optional[float] = None
    last_slope: Optional[float] = None
    for i in range(1, len(offsets)):
        delta_offset = offsets[i] - offsets[i - 1]
        delta_s = s_vals[i] - s_vals[i - 1]
        if delta_offset <= 1e-9:
            continue
        slope = delta_s / delta_offset
        if first_slope is None:
            first_slope = slope
        last_slope = slope

    if first_slope is None:
        first_slope = 1.0
    if last_slope is None:
        last_slope = first_slope

    def mapper(value: float) -> float:
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except Exception:  # pragma: no cover - defensive
                return s_vals[0]

        if value <= offsets[0]:
            return float(s_vals[0] + (value - offsets[0]) * first_slope)

        for i in range(1, len(offsets)):
            lo = offsets[i - 1]
            hi = offsets[i]
            if value <= hi:
                if hi <= lo:
                    return s_vals[i]
                t = (value - lo) / (hi - lo)
                return s_vals[i - 1] + t * (s_vals[i] - s_vals[i - 1])

        return float(s_vals[-1] + (value - offsets[-1]) * last_slope)

    return mapper


def build_elevation_profile(
    df_line_geo: DataFrame,
    *,
    offset_mapper: Optional[Callable[[float], float]] = None,
) -> List[dict]:
    """Build an OpenDRIVE elevation profile from line geometry heights.

    The PROFILETYPE_MPU_LINE_GEOMETRY.csv source encodes longitudinal
    offsets in centimetres alongside absolute height values.  The
    resulting profile is emitted as a list of dictionaries ready to be
    serialised into ``<elevation>`` elements where ``a`` represents the
    height at ``s`` and ``b`` encodes the gradient (first derivative).
    Only the primary path is considered when multiple polylines are
    present in the input.
    """

    if df_line_geo is None or len(df_line_geo) == 0:
        return []

    offset_col = _col_like(df_line_geo, "Offset")
    height_col = _find_height_column(df_line_geo)
    if offset_col is None or height_col is None:
        return []

    path_col = _col_like(df_line_geo, "Path")
    retrans_col: Optional[str] = None
    for col in df_line_geo.columns:
        if "retrans" in col.lower():
            retrans_col = col
            break

    best_path = None
    if path_col is not None and df_line_geo[path_col].nunique(dropna=True) > 1:
        counts = {}
        path_series = df_line_geo[path_col]
        for value in path_series.to_list():
            if value is None:
                continue
            counts[value] = counts.get(value, 0) + 1
        if counts:
            best_path = max(counts, key=counts.get)

    grouped: Dict[float, List[float]] = {}
    all_heights: List[float] = []

    for idx in range(len(df_line_geo)):
        row = df_line_geo.iloc[idx]

        if retrans_col is not None:
            retrans_value = row[retrans_col]
            if isinstance(retrans_value, str):
                retrans_flag = retrans_value.strip().lower() == "true"
            else:
                retrans_flag = bool(retrans_value)
            if retrans_flag:
                continue

        if best_path is not None and path_col is not None and row[path_col] != best_path:
            continue

        offset_raw = row[offset_col]
        height_raw = row[height_col]
        try:
            offset_cm = float(offset_raw)
            height = float(height_raw)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(height):
            continue

        if abs(height) >= 1e4:
            # Sentinel-style placeholders in some datasets use extremely large
            # values (for example ``83886.07``) to signal that the real
            # elevation measurement is unavailable.  Treat them as missing data
            # instead of letting them skew the typical height statistics.
            continue

        grouped.setdefault(offset_cm, []).append(height)
        all_heights.append(height)

    if not grouped:
        return []

    typical_height: Optional[float] = None
    if all_heights:
        try:
            typical_height = statistics.median(all_heights)
        except statistics.StatisticsError:  # pragma: no cover - defensive guard
            typical_height = None

    origin_cm = min(grouped.keys())

    points: List[Tuple[float, float]] = []
    for offset_cm in sorted(grouped.keys()):
        heights = grouped[offset_cm]
        if not heights:
            continue

        filtered: List[float] = []
        for value in heights:
            if not math.isfinite(value):
                continue

            if typical_height is not None:
                deviation = abs(value - typical_height)
                allowed = max(50.0, abs(typical_height) * 5.0)
                if deviation > allowed:
                    continue

            filtered.append(value)

        if not filtered:
            continue

        avg_height = sum(filtered) / len(filtered)
        offset_m = max(0.0, (offset_cm - origin_cm) * 0.01)
        if offset_mapper is not None:
            s_val = float(offset_mapper(offset_m))
        else:
            s_val = float(offset_m)
        points.append((s_val, avg_height))

    if not points:
        return []

    profile: List[dict] = []
    for idx, (s_val, height) in enumerate(points):
        if idx < len(points) - 1:
            next_s, next_height = points[idx + 1]
            if next_s > s_val:
                slope = (next_height - height) / (next_s - s_val)
            else:
                slope = 0.0
        else:
            # The final elevation entry does not have a following point to
            # constrain its gradient.  Re-using the previous slope may cause
            # the profile to extrapolate aggressively which manifests as a
            # vertical spike at the end of the road.  Default to a flat
            # continuation instead.
            slope = 0.0

        profile.append({
            "s": s_val,
            "a": height,
            "b": slope,
            "c": 0.0,
            "d": 0.0,
        })

    return profile


def _select_best_path(df: DataFrame, path_col: Optional[str]) -> Optional[Any]:
    if path_col is None:
        return None

    counts: Dict[Any, int] = {}
    series = df[path_col]
    for value in series.to_list():
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1

    if not counts:
        return None

    return max(counts, key=counts.get)


def _prepare_segment_key(start_m: float, end_m: float) -> Tuple[float, float]:
    return (round(float(start_m), 4), round(float(end_m), 4))


def build_curvature_profile(
    df_curvature: Optional[DataFrame],
    *,
    offset_mapper: Optional[Callable[[float], float]] = None,
    centerline: Optional[DataFrame] = None,
    geo_origin: Optional[Tuple[float, float]] = None,
    lane_geometry_df: Optional[DataFrame] = None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    if df_curvature is None or len(df_curvature) == 0:
        return [], []

    start_col = _find_column(df_curvature, "offset", exclude=("end",))
    end_col = _find_column(df_curvature, "end", "offset")
    curvature_col = (
        _find_column(df_curvature, "曲率", exclude=("レーン", "lane", "情報"))
        or _find_column(df_curvature, "曲率", "値")
        or _find_column(df_curvature, "曲率", "rad/m")
        or _find_column(df_curvature, "curvature", exclude=("lane", "count"))
        or _find_column(df_curvature, "curvature", "value")
        or _find_column(df_curvature, "curvature")
    )
    path_col = _find_column(df_curvature, "path")
    retrans_col = _find_column(df_curvature, "is", "retransmission")
    lane_col = (
        _find_column(df_curvature, "lane", "number")
        or _find_column(df_curvature, "lane", "no")
        or _find_column(df_curvature, "lane")
    )
    shape_col = (
        _find_column(df_curvature, "形状", "インデックス")
        or _find_column(df_curvature, "shape", "index")
        or _find_column(df_curvature, "index")
    )
    lat_col = (
        _find_column(df_curvature, "緯度")
        or _find_column(df_curvature, "latitude")
        or _find_column(df_curvature, "lat")
    )
    lon_col = (
        _find_column(df_curvature, "経度")
        or _find_column(df_curvature, "longitude")
        or _find_column(df_curvature, "lon")
    )

    if start_col is None or end_col is None or curvature_col is None:
        return [], []

    def _legacy_profile() -> List[Dict[str, float]]:
        best_path = _select_best_path(df_curvature, path_col)

        entries: List[Tuple[float, float, float]] = []
        origin_cm: Optional[float] = None

        for idx in range(len(df_curvature)):
            row = df_curvature.iloc[idx]

            if retrans_col is not None:
                retrans_val = str(row[retrans_col]).strip().lower()
                if retrans_val == "true":
                    continue

            if best_path is not None and path_col is not None and row[path_col] != best_path:
                continue

            start_cm = _to_float(row[start_col])
            end_cm = _to_float(row[end_col])
            curvature_val = _to_float(row[curvature_col])

            if start_cm is None or end_cm is None or curvature_val is None:
                continue

            if origin_cm is None or start_cm < origin_cm:
                origin_cm = start_cm

            entries.append((start_cm, end_cm, curvature_val))

        if origin_cm is None:
            return []

        grouped: Dict[Tuple[float, float], Dict[str, List[float]]] = {}

        for start_cm, end_cm, curvature_val in entries:
            start_m = max(0.0, (start_cm - origin_cm) * 0.01)
            end_m = max(0.0, (end_cm - origin_cm) * 0.01)
            if end_m <= start_m:
                continue

            key = _prepare_segment_key(start_m, end_m)
            bucket = grouped.setdefault(key, {"curvature": [], "length": []})
            bucket["curvature"].append(curvature_val)
            bucket["length"].append(end_m - start_m)

        if not grouped:
            return []

        profile: List[Dict[str, float]] = []
        for (start_m, end_m), values in sorted(grouped.items(), key=lambda item: item[0]):
            curv_values = values["curvature"]
            if not curv_values:
                continue
            avg_curvature = sum(curv_values) / len(curv_values)
            s0 = float(offset_mapper(start_m)) if offset_mapper is not None else float(start_m)
            s1 = float(offset_mapper(end_m)) if offset_mapper is not None else float(end_m)
            if s1 <= s0:
                continue
            if values["length"]:
                avg_length = sum(values["length"]) / len(values["length"])
            else:
                avg_length = end_m - start_m
            if avg_length <= 0:
                continue
            span = s1 - s0
            if span <= 0:
                continue
            scale = avg_length / span
            profile.append({"s0": s0, "s1": s1, "curvature": avg_curvature * scale})

        return profile

    if shape_col is None or lane_col is None:
        legacy = _legacy_profile()
        return legacy, []

    best_path = _select_best_path(df_curvature, path_col)

    def _normalise_key(value: Any) -> Optional[Any]:
        canonical = _canonical_numeric(value, allow_negative=True)
        if canonical is not None:
            return canonical
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    lane_origins: Dict[Tuple[Optional[Any], Optional[Any]], float] = {}
    groups: Dict[Tuple[Optional[Any], Optional[Any], float, float], Dict[str, Any]] = {}
    lane_segments: Dict[Tuple[Optional[Any], Optional[Any]], List[Dict[str, float]]] = {}
    lane_stats: Dict[Tuple[Optional[Any], Optional[Any]], Dict[str, float]] = {}
    lane_samples: Dict[Tuple[Optional[Any], Optional[Any]], List[Dict[str, Any]]] = {}

    lane_geometry_lookup: Dict[
        Tuple[Optional[Any], Optional[Any], int], Tuple[float, float]
    ] = {}

    if lane_geometry_df is not None and len(lane_geometry_df) > 0:
        geom_path_col = _find_column(lane_geometry_df, "path", "id")
        geom_lane_col = (
            _find_column(lane_geometry_df, "lane", "number")
            or _find_column(lane_geometry_df, "lane", "no")
        )
        geom_shape_col = _find_column(lane_geometry_df, "形状", "インデックス")
        geom_lat_col = _find_column(lane_geometry_df, "緯度") or _find_column(
            lane_geometry_df, "latitude"
        )
        geom_lon_col = _find_column(lane_geometry_df, "経度") or _find_column(
            lane_geometry_df, "longitude"
        )
        geom_retrans_col = _find_column(
            lane_geometry_df, "is", "retransmission"
        )

        if (
            geom_path_col is not None
            and geom_lane_col is not None
            and geom_shape_col is not None
            and geom_lat_col is not None
            and geom_lon_col is not None
        ):
            for idx in range(len(lane_geometry_df)):
                row = lane_geometry_df.iloc[idx]

                if geom_retrans_col is not None:
                    retrans_val = str(row[geom_retrans_col]).strip().lower()
                    if retrans_val == "true":
                        continue

                path_key = _normalise_key(row[geom_path_col])
                lane_key = _normalise_key(row[geom_lane_col])
                shape_val = _to_float(row[geom_shape_col])
                lat_val = _to_float(row[geom_lat_col])
                lon_val = _to_float(row[geom_lon_col])

                if (
                    lane_key is None
                    or shape_val is None
                    or lat_val is None
                    or lon_val is None
                    or not math.isfinite(lat_val)
                    or not math.isfinite(lon_val)
                ):
                    continue

                shape_idx_key = int(round(shape_val))
                key = (path_key, lane_key, shape_idx_key)
                lane_geometry_lookup.setdefault(key, (lat_val, lon_val))

    for idx in range(len(df_curvature)):
        row = df_curvature.iloc[idx]

        if retrans_col is not None:
            retrans_val = str(row[retrans_col]).strip().lower()
            if retrans_val == "true":
                continue

        raw_path = row[path_col] if path_col is not None else None
        path_key = _normalise_key(raw_path)
        if best_path is not None:
            best_key = _normalise_key(best_path)
            if path_key != best_key:
                continue
            path_key = best_key

        lane_key = _normalise_key(row[lane_col])
        if lane_key is None:
            continue

        start_cm = _to_float(row[start_col])
        end_cm = _to_float(row[end_col])
        curvature_val = _to_float(row[curvature_col])
        shape_val = _to_float(row[shape_col])
        lat_val = _to_float(row[lat_col]) if lat_col is not None else None
        lon_val = _to_float(row[lon_col]) if lon_col is not None else None

        if (lat_val is None or lon_val is None) and lane_geometry_lookup:
            shape_idx_key = None
            if shape_val is not None and math.isfinite(shape_val):
                shape_idx_key = int(round(shape_val))
            if shape_idx_key is not None:
                lookup_key = (path_key, lane_key, shape_idx_key)
                mapped = lane_geometry_lookup.get(lookup_key)
                if mapped is None and path_key is None:
                    # fall back to lane-only match for datasets that omit path IDs
                    mapped = lane_geometry_lookup.get((None, lane_key, shape_idx_key))
                if mapped is not None:
                    lat_val, lon_val = mapped

        if (
            start_cm is None
            or end_cm is None
            or curvature_val is None
            or shape_val is None
        ):
            continue

        lane_origin_key = (path_key, lane_key)
        current_origin = lane_origins.get(lane_origin_key)
        if current_origin is None or start_cm < current_origin:
            lane_origins[lane_origin_key] = start_cm

        segment_key = (
            path_key,
            lane_key,
            float(start_cm),
            float(end_cm),
        )
        bucket = groups.setdefault(segment_key, {"shapes": {}})

        shape_idx = float(shape_val)
        try:
            curvature_val_f = float(curvature_val)
        except (TypeError, ValueError):
            continue

        entry = bucket["shapes"].setdefault(
            shape_idx,
            {
                "sum": 0.0,
                "count": 0,
                "lat_sum": 0.0,
                "lon_sum": 0.0,
                "geo_count": 0,
            },
        )
        entry["sum"] += curvature_val_f
        entry["count"] += 1
        if (
            lat_val is not None
            and lon_val is not None
            and math.isfinite(lat_val)
            and math.isfinite(lon_val)
        ):
            entry["lat_sum"] += float(lat_val)
            entry["lon_sum"] += float(lon_val)
            entry["geo_count"] += 1

    if not groups:
        return [], []

    profile: List[Dict[str, float]] = []

    def _sort_key(item: Tuple[Tuple[Optional[Any], Optional[Any], float, float], Dict[str, Any]]):
        path_key, lane_key, start_cm, end_cm = item[0]
        path_text = "" if path_key is None else str(path_key)
        lane_text = "" if lane_key is None else str(lane_key)
        return (path_text, lane_text, start_cm, end_cm)

    for (path_key, lane_key, start_cm, end_cm), bucket in sorted(
        groups.items(),
        key=_sort_key,
    ):
        origin_cm = lane_origins.get((path_key, lane_key))
        if origin_cm is None:
            continue

        start_m = max(0.0, (start_cm - origin_cm) * 0.01)
        end_m = max(0.0, (end_cm - origin_cm) * 0.01)
        if end_m <= start_m:
            continue

        shapes = bucket.get("shapes", {})
        averaged_shapes: List[Tuple[float, float, Optional[float], Optional[float]]] = []
        for idx_val, stats in shapes.items():
            total = float(stats.get("sum", 0.0))
            count = float(stats.get("count", 0.0))
            if count <= 0 or not math.isfinite(total):
                continue
            averaged = total / count
            if not math.isfinite(averaged):
                continue
            geo_count = float(stats.get("geo_count", 0.0))
            if geo_count > 0:
                lat_avg = float(stats.get("lat_sum", 0.0) / geo_count)
                lon_avg = float(stats.get("lon_sum", 0.0) / geo_count)
            else:
                lat_avg = None
                lon_avg = None
            averaged_shapes.append((float(idx_val), averaged, lat_avg, lon_avg))

        if len(averaged_shapes) < 2:
            continue

        ordered = sorted(averaged_shapes, key=lambda item: item[0])
        idx_min = ordered[0][0]
        idx_max = ordered[-1][0]
        span_idx = idx_max - idx_min
        if abs(span_idx) <= 1e-12:
            continue

        bucket_segments: List[Dict[str, float]] = []
        offsets: List[float] = []
        mapped_s: List[float] = []

        lat_vals: List[float] = []
        lon_vals: List[float] = []
        has_latlon = True
        for _, _, lat_avg, lon_avg in ordered:
            if lat_avg is None or lon_avg is None:
                has_latlon = False
                break
            lat_vals.append(float(lat_avg))
            lon_vals.append(float(lon_avg))

        xy_vals: Optional[Tuple[List[float], List[float]]] = None
        if has_latlon and len(lat_vals) >= 2:
            try:
                if geo_origin is not None:
                    lat0_use, lon0_use = float(geo_origin[0]), float(geo_origin[1])
                else:
                    lat0_use, lon0_use = lat_vals[0], lon_vals[0]
                xy_vals = latlon_to_local_xy(lat_vals, lon_vals, lat0_use, lon0_use)
            except Exception:  # pragma: no cover - defensive
                xy_vals = None
                has_latlon = False
        else:
            has_latlon = False

        cumulative: List[float] = []
        if has_latlon and xy_vals is not None:
            xs, ys = xy_vals
            cumulative.append(0.0)
            for i in range(len(xs) - 1):
                dist = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
                cumulative.append(cumulative[-1] + dist)
        else:
            cumulative = []

        dataset_span_total = end_m - start_m
        if dataset_span_total <= 0:
            continue

        if cumulative and len(cumulative) == len(ordered):
            geom_length = cumulative[-1]
            if geom_length > 0:
                scale_factor = dataset_span_total / geom_length
            else:
                scale_factor = 1.0
            for dist in cumulative:
                offset_val = start_m + dist * scale_factor
                offsets.append(offset_val)
                mapped = float(offset_mapper(offset_val)) if offset_mapper is not None else float(offset_val)
                mapped_s.append(mapped)
        else:
            for idx_val, _, _, _ in ordered:
                frac = (idx_val - idx_min) / span_idx
                offset_val = start_m + dataset_span_total * frac
                offsets.append(offset_val)
                mapped = float(offset_mapper(offset_val)) if offset_mapper is not None else float(offset_val)
                mapped_s.append(mapped)

        if len(offsets) != len(ordered) or len(mapped_s) != len(ordered):
            continue

        xs_for_samples: Optional[List[float]] = None
        ys_for_samples: Optional[List[float]] = None
        if xy_vals is not None and len(xy_vals[0]) == len(ordered):
            xs_for_samples = list(xy_vals[0])
            ys_for_samples = list(xy_vals[1])

        lane_token = (path_key, lane_key)
        sample_bucket = lane_samples.setdefault(lane_token, [])

        bucket_segments: List[Dict[str, float]] = []
        resampled = None
        preserve_targets: Optional[List[float]] = None
        if cumulative and len(cumulative) == len(ordered):
            preserve_targets = []
            last_shape: Optional[float] = None
            for (shape_idx, *_), arc_val in zip(ordered, cumulative):
                if last_shape is None or shape_idx != last_shape:
                    preserve_targets.append(float(arc_val))
                    last_shape = shape_idx
            if cumulative:
                last_arc = float(cumulative[-1])
                if not preserve_targets or abs(preserve_targets[-1] - last_arc) > 1e-9:
                    preserve_targets.append(last_arc)
        if (
            xs_for_samples is not None
            and ys_for_samples is not None
            and len(xs_for_samples) >= 2
            and len(ys_for_samples) >= 2
        ):
            resampled = _resample_parametric_curve(
                mapped_s,
                xs_for_samples,
                ys_for_samples,
                step=CURVATURE_RESAMPLE_STEP,
                preserve_targets=preserve_targets,
            )

        if resampled is not None and len(resampled.get("s", [])) >= 2:
            s_series = [float(val) for val in resampled["s"]]
            curvature_series = [float(val) for val in resampled["curvature"]]
            x_series = [float(val) for val in resampled.get("x", [])]
            y_series = [float(val) for val in resampled.get("y", [])]
            offset_series = _interp_values(mapped_s, offsets, s_series) if offsets else [0.0 for _ in s_series]
            shape_series = _interp_values(
                mapped_s,
                [item[0] for item in ordered],
                s_series,
            )

            for idx_sample, s_val in enumerate(s_series):
                if idx_sample >= len(curvature_series):
                    break
                offset_val = offset_series[idx_sample] if idx_sample < len(offset_series) else offset_series[-1]
                shape_val = shape_series[idx_sample] if idx_sample < len(shape_series) else None
                sample_entry: Dict[str, Any] = {
                    "s": float(s_val),
                    "offset": float(offset_val),
                    "curvature": float(curvature_series[idx_sample]),
                    "path": path_key,
                    "lane": lane_key,
                }
                if shape_val is not None and math.isfinite(shape_val):
                    sample_entry["shape_index"] = float(shape_val)
                if idx_sample < len(x_series) and idx_sample < len(y_series):
                    sample_entry["x"] = float(x_series[idx_sample])
                    sample_entry["y"] = float(y_series[idx_sample])
                sample_bucket.append(sample_entry)

            for idx_seg in range(len(s_series) - 1):
                s0 = s_series[idx_seg]
                s1 = s_series[idx_seg + 1]
                if not (
                    math.isfinite(s0)
                    and math.isfinite(s1)
                    and s1 > s0
                ):
                    continue
                c0 = curvature_series[idx_seg]
                c1 = curvature_series[idx_seg + 1]
                curvature_val = 0.5 * (c0 + c1)
                if not math.isfinite(curvature_val):
                    continue
                bucket_segments.append({"s0": float(s0), "s1": float(s1), "curvature": float(curvature_val)})
        else:
            for i, (idx_val, curv_val, _, _) in enumerate(ordered):
                if i >= len(offsets):
                    break
                offset_val = offsets[i]
                mapped_val = mapped_s[i]
                sample_entry = {
                    "s": float(mapped_val),
                    "offset": float(offset_val),
                    "curvature": float(curv_val),
                    "path": path_key,
                    "lane": lane_key,
                    "shape_index": float(idx_val),
                }
                if (
                    xs_for_samples is not None
                    and ys_for_samples is not None
                    and i < len(xs_for_samples)
                ):
                    sample_entry["x"] = float(xs_for_samples[i])
                    sample_entry["y"] = float(ys_for_samples[i])
                sample_bucket.append(sample_entry)

            for i in range(len(ordered) - 1):
                idx0, curv0, _, _ = ordered[i]
                idx1, _, _, _ = ordered[i + 1]
                if idx1 <= idx0:
                    continue

                offset0 = offsets[i]
                offset1 = offsets[i + 1]
                s0 = mapped_s[i]
                s1 = mapped_s[i + 1]

                if not (
                    math.isfinite(offset0)
                    and math.isfinite(offset1)
                    and math.isfinite(s0)
                    and math.isfinite(s1)
                ):
                    continue
                if offset1 <= offset0 or s1 <= s0:
                    continue

                dataset_span = offset1 - offset0
                if dataset_span <= 0:
                    continue

                s_span = s1 - s0
                if s_span <= 0:
                    continue

                curv0_val = float(curv0)
                if not math.isfinite(curv0_val):
                    continue

                scale = dataset_span / s_span if s_span > 1e-12 else 1.0
                curvature_val = curv0_val * scale

                if not math.isfinite(curvature_val):
                    continue

                bucket_segments.append({"s0": s0, "s1": s1, "curvature": curvature_val})

        if not bucket_segments:
            continue

        if centerline is not None:
            start_s = bucket_segments[0]["s0"]
            end_s = bucket_segments[-1]["s1"]
            _, _, start_hdg = _interpolate_centerline(centerline, start_s)
            _, _, end_hdg = _interpolate_centerline(centerline, end_s)
            target_delta = _normalize_angle(end_hdg - start_hdg)
            dataset_delta = 0.0
            total_span = 0.0
            for seg in bucket_segments:
                span = seg["s1"] - seg["s0"]
                dataset_delta += seg["curvature"] * span
                total_span += span
            if abs(dataset_delta) > 1e-12 and math.isfinite(dataset_delta):
                scale_factor = target_delta / dataset_delta if math.isfinite(target_delta) else 1.0
                if math.isfinite(scale_factor):
                    for seg in bucket_segments:
                        seg["curvature"] *= scale_factor
            elif total_span > 1e-9:
                uniform_curv = target_delta / total_span
                for seg in bucket_segments:
                    seg["curvature"] = uniform_curv

        lane_token = (path_key, lane_key)
        stats = lane_stats.setdefault(
            lane_token,
            {"coverage": 0.0, "samples": 0.0},
        )
        for segment in bucket_segments:
            lane_segments.setdefault(lane_token, []).append(segment)
            stats["coverage"] += segment["s1"] - segment["s0"]
            stats["samples"] += 1.0

    if not lane_segments:
        return [], []

    def _lane_priority(
        lane_key: Optional[Any],
        stats: Dict[str, float],
    ) -> Tuple[float, float, float, float, str]:
        coverage = float(stats.get("coverage", 0.0))
        samples = float(stats.get("samples", 0.0))
        numeric_value = None
        try:
            if isinstance(lane_key, (int, float)) and math.isfinite(float(lane_key)):
                numeric_value = float(lane_key)
        except Exception:  # pragma: no cover - defensive
            numeric_value = None

        if numeric_value is not None:
            proximity = abs(numeric_value)
            signed_value = numeric_value
        else:
            proximity = float("inf")
            signed_value = float("inf")

        lane_text = "" if lane_key is None else str(lane_key)
        return (-coverage, -samples, proximity, signed_value, lane_text)

    segments_by_path: Dict[Optional[Any], List[Tuple[Optional[Any], List[Dict[str, float]]]]] = {}
    for (path_key, lane_key), segments in lane_segments.items():
        segments_by_path.setdefault(path_key, []).append((lane_key, segments))

    selected_samples: List[Dict[str, Any]] = []

    for path_key, lane_entries in segments_by_path.items():
        if not lane_entries:
            continue

        best_lane_key, best_segments = min(
            (
                (lane_key, segments)
                for lane_key, segments in lane_entries
            ),
            key=lambda item: _lane_priority(item[0], lane_stats.get((path_key, item[0]), {})),
        )

        ordered_segments = sorted(best_segments, key=lambda seg: seg["s0"])
        profile.extend(ordered_segments)

        lane_token = (path_key, best_lane_key)
        sample_values = lane_samples.get(lane_token)
        if sample_values:
            sample_values = sorted(
                sample_values,
                key=lambda item: (float(item.get("s", 0.0)), float(item.get("offset", 0.0))),
            )
            selected_samples.extend(sample_values)

    profile.sort(key=lambda item: item["s0"])
    return profile, selected_samples


def build_slope_profile(
    df_slope: Optional[DataFrame],
    *,
    offset_mapper: Optional[Callable[[float], float]] = None,
) -> Dict[str, List[Dict[str, float]]]:
    if df_slope is None or len(df_slope) == 0:
        return {"longitudinal": [], "superelevation": []}

    start_col = _find_column(df_slope, "offset", exclude=("end",))
    end_col = _find_column(df_slope, "end", "offset")
    slope_col = _find_column(df_slope, "縦断勾配") or _find_column(df_slope, "longitudinal", "%")
    cross_col = _find_column(df_slope, "横断勾配") or _find_column(df_slope, "cross", "%")
    path_col = _find_column(df_slope, "path")
    retrans_col = _find_column(df_slope, "is", "retransmission")

    if start_col is None or end_col is None:
        return {"longitudinal": [], "superelevation": []}

    best_path = _select_best_path(df_slope, path_col)

    entries: List[Tuple[float, float, Optional[float], Optional[float]]] = []
    origin_cm: Optional[float] = None

    for idx in range(len(df_slope)):
        row = df_slope.iloc[idx]

        if retrans_col is not None:
            retrans_val = str(row[retrans_col]).strip().lower()
            if retrans_val == "true":
                continue

        if best_path is not None and path_col is not None and row[path_col] != best_path:
            continue

        start_cm = _to_float(row[start_col])
        end_cm = _to_float(row[end_col])
        if start_cm is None or end_cm is None:
            continue

        if origin_cm is None or start_cm < origin_cm:
            origin_cm = start_cm

        grade_val = _to_float(row[slope_col]) if slope_col is not None else None
        cross_val = _to_float(row[cross_col]) if cross_col is not None else None

        entries.append((start_cm, end_cm, grade_val, cross_val))

    if origin_cm is None:
        return {"longitudinal": [], "superelevation": []}

    grouped: Dict[Tuple[float, float], Dict[str, List[float]]] = {}

    for start_cm, end_cm, grade_val, cross_val in entries:
        start_m = max(0.0, (start_cm - origin_cm) * 0.01)
        end_m = max(0.0, (end_cm - origin_cm) * 0.01)
        if end_m <= start_m:
            continue

        entry = grouped.setdefault(
            _prepare_segment_key(start_m, end_m),
            {"grade": [], "cross": [], "length": []},
        )

        if grade_val is not None:
            entry["grade"].append(grade_val * 0.01)

        if cross_val is not None:
            entry["cross"].append(cross_val * 0.01)

        entry["length"].append(end_m - start_m)

    longitudinal: List[Dict[str, float]] = []
    superelevation: List[Dict[str, float]] = []

    for (start_m, end_m), values in sorted(grouped.items(), key=lambda item: item[0]):
        s0 = float(offset_mapper(start_m)) if offset_mapper is not None else float(start_m)
        s1 = float(offset_mapper(end_m)) if offset_mapper is not None else float(end_m)
        if s1 <= s0:
            continue

        if values["grade"]:
            avg_grade = sum(values["grade"]) / len(values["grade"])
            segment_span = s1 - s0
            if segment_span <= 0:
                continue
            if values["length"]:
                avg_length = sum(values["length"]) / len(values["length"])
            else:
                avg_length = segment_span
            if avg_length > 0:
                scale = avg_length / segment_span
                avg_grade *= scale
            longitudinal.append({"s0": s0, "s1": s1, "grade": avg_grade})

        if values["cross"]:
            avg_cross = sum(values["cross"]) / len(values["cross"])
            superelevation.append({"s0": s0, "s1": s1, "angle": avg_cross})

    return {"longitudinal": longitudinal, "superelevation": superelevation}
def build_elevation_profile_from_slopes(
    segments: List[Dict[str, float]],
    *,
    initial_height: float = 0.0,
) -> List[Dict[str, float]]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: (item["s0"], item.get("s1", item["s0"])))

    profile: List[Dict[str, float]] = []
    height = float(initial_height)
    prev_end = ordered[0]["s0"]
    last_grade = ordered[0]["grade"]

    first_s0 = ordered[0]["s0"]
    profile.append({"s": first_s0, "a": height, "b": last_grade, "c": 0.0, "d": 0.0})

    prev_end = ordered[0].get("s1", first_s0)
    if prev_end > first_s0:
        height += last_grade * (prev_end - first_s0)

    for seg in ordered[1:]:
        s0 = seg["s0"]
        s1 = seg.get("s1", s0)
        grade = seg["grade"]

        if s0 > prev_end:
            height += last_grade * (s0 - prev_end)
            prev_end = s0
        elif s0 < prev_end:
            s0 = prev_end

        profile.append({"s": s0, "a": height, "b": grade, "c": 0.0, "d": 0.0})

        if s1 > prev_end:
            height += grade * (s1 - s0)
            prev_end = s1

        last_grade = grade

    return profile


def build_superelevation_profile(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: (item["s0"], item.get("s1", item["s0"])))
    profile: List[Dict[str, float]] = []
    for seg in ordered:
        profile.append({"s": seg["s0"], "a": seg["angle"], "b": 0.0, "c": 0.0, "d": 0.0})
    return profile


def _interpolate_centerline(centerline: DataFrame, target_s: float) -> Tuple[float, float, float]:
    s_vals = [float(v) for v in centerline["s"].to_list()]
    x_vals = [float(v) for v in centerline["x"].to_list()]
    y_vals = [float(v) for v in centerline["y"].to_list()]
    hdg_vals = [float(v) for v in centerline["hdg"].to_list()]

    if not s_vals:
        return 0.0, 0.0, 0.0

    if target_s <= s_vals[0]:
        return x_vals[0], y_vals[0], hdg_vals[0]

    for idx in range(1, len(s_vals)):
        s_prev = s_vals[idx - 1]
        s_curr = s_vals[idx]
        if target_s <= s_curr:
            span = s_curr - s_prev
            if span <= 0:
                return x_vals[idx], y_vals[idx], hdg_vals[idx]
            t = (target_s - s_prev) / span
            x = x_vals[idx - 1] + t * (x_vals[idx] - x_vals[idx - 1])
            y = y_vals[idx - 1] + t * (y_vals[idx] - y_vals[idx - 1])
            if abs(target_s - s_curr) <= 1e-6:
                hdg = hdg_vals[min(idx, len(hdg_vals) - 1)]
            else:
                hdg = hdg_vals[idx - 1]
            return x, y, hdg

    return x_vals[-1], y_vals[-1], hdg_vals[-1]


def _suppress_curvature_spikes(
    segments: List[Dict[str, float]],
    *,
    min_span: float = 0.4,
    spike_curvature: float = 0.01,
    curvature_similarity: float = 0.15,
) -> List[Dict[str, float]]:
    """Collapse very short curvature spans that oscillate in sign.

    Shape-index based curvature samples occasionally encode centimetre-long
    arcs whose curvature abruptly flips sign relative to the surrounding
    segments.  Offsetting these spikes to generate lane boundaries amplifies
    the oscillation and renders as visibly jagged white lines.  The source
    geometry, however, remains smooth when the offending spikes are replaced
    by their longer neighbours.

    This helper trims suspicious spans before the planView approximation is
    constructed so that downstream integration only sees curvature that varies
    gradually along the reference line.
    """

    if not segments:
        return []

    # Normalise the entries to avoid mutating the caller's data while keeping
    # track of any auxiliary metadata that may be present on each segment.
    ordered: List[Dict[str, float]] = []
    for raw in segments:
        try:
            start = float(raw["s0"])
            end = float(raw["s1"])
        except (KeyError, TypeError, ValueError):
            continue

        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if end - start <= 1e-9:
            continue

        entry = dict(raw)
        entry["s0"] = start
        entry["s1"] = end

        curvature_raw = raw.get("curvature", 0.0)
        try:
            curvature = float(curvature_raw)
        except (TypeError, ValueError):
            curvature = 0.0
        if not math.isfinite(curvature):
            curvature = 0.0
        entry["curvature"] = curvature
        ordered.append(entry)

    if not ordered:
        return []

    ordered.sort(key=lambda seg: seg["s0"])

    def _segment_span(entry: Dict[str, float]) -> float:
        return float(entry["s1"]) - float(entry["s0"])

    def _curvature_info(entry: Optional[Dict[str, float]]) -> Tuple[Optional[float], float]:
        if entry is None:
            return None, 0.0
        value = float(entry.get("curvature", 0.0))
        if abs(value) <= 1e-12:
            return None, value
        return math.copysign(1.0, value), value

    changed = True
    while changed:
        changed = False
        for idx in range(1, len(ordered) - 1):
            current = ordered[idx]
            span = _segment_span(current)
            curvature = float(current.get("curvature", 0.0))
            if span >= min_span - 1e-9:
                continue
            if abs(curvature) < spike_curvature:
                continue

            prev_entry = ordered[idx - 1]
            next_entry = ordered[idx + 1]
            prev_sign, prev_curv = _curvature_info(prev_entry)
            next_sign, next_curv = _curvature_info(next_entry)
            curr_sign, _ = _curvature_info(current)

            if curr_sign is None:
                continue
            if prev_sign is None or next_sign is None:
                continue
            if prev_sign != next_sign:
                continue
            if curr_sign == prev_sign:
                continue

            reference = max(abs(prev_curv), abs(next_curv), spike_curvature)
            if abs(prev_curv - next_curv) > curvature_similarity * reference:
                continue

            prev_entry["s1"] = max(prev_entry["s1"], current["s1"])
            if next_entry["s0"] < prev_entry["s1"]:
                next_entry["s0"] = prev_entry["s1"]

            del ordered[idx]
            changed = True
            break

    cleaned: List[Dict[str, float]] = []
    for seg in ordered:
        span = _segment_span(seg)
        if span <= 1e-9:
            continue
        cleaned.append(seg)

    return cleaned


def build_geometry_segments(
    centerline: DataFrame,
    curvature_segments: List[Dict[str, float]],
    *,
    max_endpoint_deviation: float = 0.5,
    max_segment_length: float = 2.0,
) -> List[Dict[str, float]]:
    if not curvature_segments:
        return []

    curvature_segments = _suppress_curvature_spikes(list(curvature_segments))
    if not curvature_segments:
        return []

    total_length = float(centerline["s"].iloc[-1]) if len(centerline) else 0.0
    if total_length <= 0:
        return []

    centerline_s_raw = [float(v) for v in centerline["s"].to_list()]

    min_anchor_spacing = 0.5
    try:
        candidate_spacing = float(max_segment_length) * 0.5
    except (TypeError, ValueError):
        candidate_spacing = None
    if candidate_spacing is not None and candidate_spacing > 0:
        min_anchor_spacing = max(min_anchor_spacing, candidate_spacing)

    centerline_s: List[float] = []
    for value in centerline_s_raw:
        if not centerline_s:
            centerline_s.append(value)
            continue

        if value - centerline_s[-1] < min_anchor_spacing:
            continue

        centerline_s.append(value)

    if centerline_s_raw:
        last_value = centerline_s_raw[-1]
        if not centerline_s or abs(centerline_s[-1] - last_value) > 1e-9:
            centerline_s.append(last_value)

    def _build_segments(max_len: float) -> Tuple[List[Dict[str, float]], float]:
        def _clamp(value: float) -> float:
            clamped = max(0.0, min(total_length, float(value)))
            # 标记点在浮点运算中会出现微小误差，这里统一取 9 位小数以便后续
            # 查找时能够复用相同的键，避免 densify 过程中产生的重复点无法被
            # 正确识别为同一个位置。
            return round(clamped, 9)

        anchor_points: Dict[float, bool] = {_clamp(0.0): True, _clamp(total_length): True}

        for value in centerline_s:
            anchor_points.setdefault(_clamp(value), True)

        if max_len <= 0:
            effective_len = 2.0
        else:
            effective_len = float(max_len)

        # 长曲率区段在积分时会累积轻微的横向偏移，进而在 OpenDRIVE
        # 查看器中表现为相邻路段之间出现细小豁口。为了在不牺牲曲线段
        # 的情况下压制误差，将曲率分段进一步按照参考线采样 densify。
        # 这里允许调用方传入更小的 densify 间距；如果输入的最大长度
        # 失效，则退回默认的 2 米。
        # densify 由后续针对每个主控制点区间的细分逻辑统一处理，这里无需
        # 额外插入中间节点。

        for seg in curvature_segments:
            s0 = _clamp(seg["s0"])
            s1 = _clamp(seg["s1"])
            anchor_points.setdefault(s0, True)
            anchor_points.setdefault(s1, True)

        ordered_points = sorted(anchor_points.keys())
        segments: List[Dict[str, float]] = []

        def _curvature_for_interval(start: float, end: float) -> float:
            mid = (start + end) / 2.0
            for seg in curvature_segments:
                if seg["s0"] <= mid <= seg["s1"]:
                    return float(seg["curvature"])
            return 0.0

        def _segment_derivatives(
            start_hdg: float,
            curvature: float,
            length: float,
        ) -> Tuple[float, float, float]:
            if abs(curvature) <= 1e-8:
                half_l_sq = 0.5 * (length ** 2)
                dxdk = -half_l_sq * math.sin(start_hdg)
                dydk = half_l_sq * math.cos(start_hdg)
                return dxdk, dydk, length

            end_hdg = start_hdg + curvature * length
            sin_start = math.sin(start_hdg)
            cos_start = math.cos(start_hdg)
            sin_end = math.sin(end_hdg)
            cos_end = math.cos(end_hdg)
            k = curvature
            L = length
            numerator_x = sin_end - sin_start
            numerator_y = cos_end - cos_start
            dxdk = (cos_end * L * k - numerator_x) / (k * k)
            dydk = (sin_end * L * k + numerator_y) / (k * k)
            return dxdk, dydk, L

        def _combined_error(
            px: float,
            py: float,
            phdg: float,
            tx: float,
            ty: float,
            thdg: float,
            span: float,
        ) -> float:
            pos_err = math.hypot(px - tx, py - ty)
            ang_err = abs(_normalize_angle(phdg - thdg))
            if ang_err <= 1e-9:
                return pos_err
            return max(pos_err, ang_err * max(1.0, span))

        def _refine_curvature(
            start_x: float,
            start_y: float,
            start_hdg: float,
            length: float,
            initial_curvature: float,
            target_x: float,
            target_y: float,
            target_hdg: float,
            preferred_curvature: float,
        ) -> Tuple[float, float, float, float, float]:
            curvature = float(initial_curvature)
            preferred_sign = 0.0
            if abs(preferred_curvature) > 1e-12:
                preferred_sign = math.copysign(1.0, preferred_curvature)
            weight_angle = max(0.25, length * length)
            for _ in range(8):
                end_x, end_y, end_hdg = _advance_pose(start_x, start_y, start_hdg, curvature, length)
                err_x = target_x - end_x
                err_y = target_y - end_y
                err_theta = _normalize_angle(target_hdg - end_hdg)
                if math.hypot(err_x, err_y) <= 1e-4 and abs(err_theta) <= 1e-4:
                    break

                dxdk, dydk, dthdk = _segment_derivatives(start_hdg, curvature, length)
                denom = dxdk * dxdk + dydk * dydk + weight_angle * (dthdk * dthdk)
                if denom <= 1e-18 or not math.isfinite(denom):
                    break

                numer = err_x * dxdk + err_y * dydk + weight_angle * err_theta * dthdk
                if not math.isfinite(numer):
                    break

                delta = numer / denom
                if not math.isfinite(delta):
                    break

                max_step = 0.5 / max(1.0, length)
                if delta > max_step:
                    delta = max_step
                elif delta < -max_step:
                    delta = -max_step

                if abs(delta) <= 1e-12:
                    break

                next_curvature = curvature + delta
                if preferred_sign and next_curvature * preferred_sign < 0:
                    next_curvature = float(preferred_curvature)
                curvature = next_curvature

            end_x, end_y, end_hdg = _advance_pose(start_x, start_y, start_hdg, curvature, length)
            endpoint_error = _combined_error(end_x, end_y, end_hdg, target_x, target_y, target_hdg, length)
            if preferred_sign and curvature * preferred_sign < 0:
                curvature = float(preferred_curvature)
                end_x, end_y, end_hdg = _advance_pose(start_x, start_y, start_hdg, curvature, length)
                endpoint_error = _combined_error(end_x, end_y, end_hdg, target_x, target_y, target_hdg, length)
            return curvature, end_x, end_y, end_hdg, endpoint_error

        current_s = ordered_points[0]
        current_x, current_y, current_hdg = _interpolate_centerline(centerline, current_s)
        continuous_x, continuous_y, continuous_hdg = current_x, current_y, current_hdg
        max_observed_endpoint_deviation = 0.0

        min_split_length = max(0.25, effective_len * 0.25)

        idx = 0
        while idx < len(ordered_points) - 1:
            start = ordered_points[idx]
            end = ordered_points[idx + 1]
            length_total = end - start
            if length_total <= 1e-6:
                idx += 1
                continue


            current_s = start

            anchor_x, anchor_y, anchor_hdg = _interpolate_centerline(centerline, start)

            drift_pos = math.hypot(anchor_x - continuous_x, anchor_y - continuous_y)
            drift_hdg = abs(_normalize_angle(anchor_hdg - continuous_hdg))

            # 形状インデックス数据的分段更密集，直接将起点强行重置到解析
            # 中心线会让上一段的数值积分结果与新的起点出现 2~3cm 的错位。
            # 当误差仍处于几十厘米以内时，保留连续积分得到的起点可以维持
            # 几何连续性；只有当累积漂移明显放大时才回退到解析中心线重新
            # 对齐，从而兼顾稳定性与准确度。
            if drift_pos > 0.1 or drift_hdg > 2e-2:
                # 直接将起点重置到解析中心线会让相邻段之间出现十几厘米的跳变。
                # 当漂移超过硬阈值时，以固定的“步长”逐渐收敛到解析位置。
                # 为了避免明显裂缝，将单次平移压缩到 1cm 以内，同时按比例
                # 压缩航向偏差，既能阻止累计误差继续扩大，又能保证输出几何
                # 的连续性。
                if drift_pos > 1e-9:
                    step_cap = max(0.005, min(0.01, length_total * 0.1))
                    max_step = min(drift_pos, step_cap)
                    factor = max_step / drift_pos
                    continuous_x += (anchor_x - continuous_x) * factor
                    continuous_y += (anchor_y - continuous_y) * factor
                delta_hdg = _normalize_angle(anchor_hdg - continuous_hdg)
                if abs(delta_hdg) > 1e-9:
                    max_turn = min(max(2e-3, length_total * 0.1), abs(delta_hdg))
                    continuous_hdg = _normalize_angle(
                        continuous_hdg + math.copysign(max_turn, delta_hdg)
                    )

                current_x, current_y, current_hdg = continuous_x, continuous_y, continuous_hdg
            else:
                # 形状インデックスの曲率は、解析中心線に対して数ミリの横ずれ
                # が頻繁に発生する。ここで過度に位置を引き戻すと前一区間との
                # 接合部に目視できる段差が残ってしまうため、角度のみを緩やか
                # に補正しつつ、位置は 3mm を超える場合に限って 1.5mm を上限
                # とする微小な補正を掛ける。これにより連続性を保ちながら徐々に
                # 漂移を抑え込める。
                if drift_pos > 0.003 and drift_pos > 1e-9:
                    step_cap = max(0.00075, min(0.0015, length_total * 0.02))
                    max_step = min(drift_pos, step_cap)
                    factor = max_step / drift_pos
                    continuous_x += (anchor_x - continuous_x) * factor
                    continuous_y += (anchor_y - continuous_y) * factor

                delta_hdg = _normalize_angle(anchor_hdg - continuous_hdg)
                if abs(delta_hdg) > 1e-6:
                    relax = max(0.05, min(0.25, length_total * 0.05))
                    continuous_hdg = _normalize_angle(continuous_hdg + delta_hdg * relax)

                current_x, current_y, current_hdg = continuous_x, continuous_y, continuous_hdg
            curvature_dataset = _curvature_for_interval(start, end)
            target_x, target_y, target_hdg = _interpolate_centerline(centerline, end)

            delta_target = _normalize_angle(target_hdg - current_hdg)
            delta_dataset = curvature_dataset * length_total
            preferred_curvature = 0.0
            preferred_sign = 0.0
            alignment_tolerance = None
            if abs(curvature_dataset) > 1e-12:
                alignment_error = abs(_normalize_angle(delta_target - delta_dataset))
                # 采样噪声会让 CSV 中的曲率与解析中心线推导出的航向变化不一致。
                # 只有当两者在一个较严格的阈值内吻合时，才继续“锁定”原始曲率，
                # 否则放宽限制让数值求解重新调整曲率，避免出现整段平移的误差。
                base_tolerance = 2e-3
                curvature_tolerance = abs(delta_dataset) * 0.25
                target_tolerance = abs(delta_target) * 0.25
                tolerance = max(base_tolerance, min(curvature_tolerance, target_tolerance))
                alignment_tolerance = tolerance
                if alignment_error <= tolerance:
                    preferred_curvature = curvature_dataset
                    preferred_sign = math.copysign(1.0, curvature_dataset)

            if abs(delta_target) > 1e-5 and length_total > 1e-6:
                curvature_guess = curvature_dataset + (delta_target - delta_dataset) / length_total
            else:
                curvature_guess = curvature_dataset

            if preferred_sign and curvature_guess * preferred_sign < 0:
                curvature_guess = preferred_curvature

            next_x, next_y, next_hdg = _advance_pose(
                current_x, current_y, current_hdg, curvature_guess, length_total
            )
            endpoint_error = _combined_error(
                next_x, next_y, next_hdg, target_x, target_y, target_hdg, length_total
            )

            def _midpoint_error(candidate_curvature: float) -> float:
                if length_total <= 1e-6:
                    return 0.0

                mid_x, mid_y, _ = _advance_pose(
                    current_x,
                    current_y,
                    current_hdg,
                    candidate_curvature,
                    length_total * 0.5,
                )
                target_mid_x, target_mid_y, _ = _interpolate_centerline(
                    centerline, (start + end) * 0.5
                )
                return math.hypot(mid_x - target_mid_x, mid_y - target_mid_y)

            midpoint_error = _midpoint_error(curvature_guess)
            segment_error = max(endpoint_error, midpoint_error)

            refined_curvature, next_x, next_y, next_hdg, endpoint_error = _refine_curvature(
                current_x,
                current_y,
                current_hdg,
                length_total,
                curvature_guess,
                target_x,
                target_y,
                target_hdg,
                preferred_curvature,
            )
            midpoint_error = _midpoint_error(refined_curvature)
            segment_error = max(endpoint_error, midpoint_error)

            if preferred_sign and refined_curvature * preferred_sign < 0:
                refined_curvature = float(preferred_curvature)
                next_x, next_y, next_hdg = _advance_pose(
                    current_x, current_y, current_hdg, refined_curvature, length_total
                )
                endpoint_error = _combined_error(
                    next_x, next_y, next_hdg, target_x, target_y, target_hdg, length_total
                )
                midpoint_error = _midpoint_error(refined_curvature)
                segment_error = max(endpoint_error, midpoint_error)

            steps = max(1, int(math.ceil(length_total / effective_len)))
            step_length = length_total / steps

            trial_segments: List[Dict[str, float]] = []
            propagated_x, propagated_y, propagated_hdg = current_x, current_y, current_hdg

            for step in range(steps):
                seg_s = start + step * step_length

                trial_segments.append(
                    {
                        "s": seg_s,
                        "x": propagated_x,
                        "y": propagated_y,
                        "hdg": propagated_hdg,
                        "length": step_length,
                        "curvature": refined_curvature,
                    }
                )

                propagated_x, propagated_y, propagated_hdg = _advance_pose(
                    propagated_x,
                    propagated_y,
                    propagated_hdg,
                    refined_curvature,
                    step_length,
                )

            need_split = (
                segment_error > max_endpoint_deviation + 1e-9
                and length_total > min_split_length + 1e-9
            )

            if need_split:
                midpoint = _clamp((start + end) * 0.5)
                if midpoint <= start + 1e-9 or end - midpoint <= 1e-9:
                    need_split = False
                else:
                    ordered_points.insert(idx + 1, midpoint)
                    continue

            segments.extend(trial_segments)

            continuous_x, continuous_y, continuous_hdg = propagated_x, propagated_y, propagated_hdg
            current_s = end

            # Record the analytical centreline pose for diagnostic purposes while
            # keeping ``continuous_*`` anchored to the numerically integrated
            # result so the emitted geometry remains contiguous.
            current_x, current_y, current_hdg = target_x, target_y, target_hdg

            if segment_error > max_observed_endpoint_deviation:
                max_observed_endpoint_deviation = segment_error

            idx += 1

        return segments, max_observed_endpoint_deviation

    try:
        initial_len = float(max_segment_length)
    except (TypeError, ValueError):
        initial_len = 2.0

    if initial_len <= 0:
        initial_len = 2.0

    try:
        max_endpoint_deviation = float(max_endpoint_deviation)
    except (TypeError, ValueError):
        max_endpoint_deviation = 0.5

    # 长距离路段若允许 2cm 以上的端点误差，会在 OpenDRIVE 查看器中留
    # 下肉眼可见的缝隙；而几米长的短路段则需要保留更宽松的阈值，否
    # 则在数据略有噪声时无法生成几何。根据参考线总长度自适应收紧
    # 阈值，可以兼顾长距离道路的连续性与单元测试所覆盖的短样例。
    if total_length >= 50.0:
        tightened = 0.01
    else:
        tightened = 0.02
    if max_endpoint_deviation > tightened:
        max_endpoint_deviation = tightened

    best_segments: List[Dict[str, float]] = []
    best_deviation = float("inf")

    def _record_best(candidate: List[Dict[str, float]], deviation_value: float) -> None:
        nonlocal best_segments, best_deviation
        if candidate and deviation_value < best_deviation:
            best_segments = candidate
            best_deviation = deviation_value

    segments, deviation = _build_segments(initial_len)
    if not segments:
        return []

    segments = _merge_geometry_segments(segments, max_segment_length=initial_len)
    _record_best(segments, deviation)

    min_len = max(0.25, initial_len / 16.0)
    current_len = initial_len

    # 在实际数据中，个别曲率段可能因为原始测量噪声导致理论圆弧与
    # 折线终点存在略高于阈值的偏差。通过逐步缩小 densify 间距可以
    # 有效抑制误差，而无需完全回退到折线表达。
    while deviation > max_endpoint_deviation and current_len > min_len:
        current_len = max(min_len, current_len / 2.0)
        candidate_segments, candidate_deviation = _build_segments(current_len)
        if not candidate_segments:
            break

        segments = _merge_geometry_segments(candidate_segments, max_segment_length=initial_len)
        deviation = candidate_deviation
        _record_best(segments, deviation)

    if deviation > max_endpoint_deviation:
        if best_segments:
            return best_segments
        return []

    return segments


def _merge_geometry_segments(
    segments: List[Dict[str, float]],
    *,
    curvature_tol: float = 1e-9,
    position_tol: float = 1e-6,
    heading_tol: float = 1e-6,
    max_segment_length: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Collapse consecutive geometry segments that share the same curvature."""

    if not segments:
        return []

    cleaned: List[Dict[str, float]] = []
    for seg in segments:
        try:
            length = float(seg.get("length", 0.0))
        except (TypeError, ValueError):
            length = 0.0
        if not math.isfinite(length) or length <= 1e-9:
            # 形状インデックス曲率在部分路段之间可能会插入零长度的
            # 补间段，用于标记数据缺口。在输出为 OpenDRIVE 几何之前
            # 需要将其过滤掉，否则查看器会把这类节点当成新的起点，
            # 继而导致整段道路出现平移错位。
            continue
        cleaned.append(dict(seg))

    if not cleaned:
        return []

    merged: List[Dict[str, float]] = []
    current = cleaned[0]
    merge_threshold = None
    if max_segment_length is not None and math.isfinite(max_segment_length):
        try:
            merge_threshold = max(0.0, float(max_segment_length))
        except (TypeError, ValueError):
            merge_threshold = None

    for seg in cleaned[1:]:
        next_seg = dict(seg)
        current_curv = float(current.get("curvature", 0.0))
        seg_curv = float(next_seg.get("curvature", 0.0))
        same_curvature = abs(seg_curv - current_curv) <= curvature_tol

        expected_s = current["s"] + current["length"]
        end_x, end_y, end_hdg = _advance_pose(
            current["x"], current["y"], current["hdg"], current_curv, current["length"]
        )

        delta_s = abs(next_seg["s"] - expected_s)
        delta_pos = math.hypot(next_seg["x"] - end_x, next_seg["y"] - end_y)
        delta_hdg = abs(_normalize_angle(next_seg["hdg"] - end_hdg))

        # Shape-index datasets may accumulate millimetre-level drift between
        # consecutive curvature spans.  Allow a slightly looser tolerance that
        # scales with the local geometry so that neighbouring segments snap back
        # to the analytically continuous pose instead of rendering as disjoint
        # road pieces in OpenDRIVE viewers.
        length_scale = max(
            float(current.get("length", 0.0)),
            float(next_seg.get("length", 0.0)),
            1e-9,
        )
        adaptive_position_tol = max(
            position_tol,
            min(0.05, 0.001 + 0.005 * length_scale),
        )
        heading_budget = abs(current_curv * current.get("length", 0.0)) + abs(
            seg_curv * next_seg.get("length", 0.0)
        )
        adaptive_heading_tol = max(
            heading_tol,
            min(0.01, 0.0025 + 0.5 * heading_budget),
        )

        if delta_s <= 1e-8:
            # When the arc-length is continuous treat the numerically
            # integrated pose as authoritative so the emitted geometry remains
            # contiguous even if the dataset introduces small lateral offsets.
            next_seg["s"] = expected_s
            next_seg["x"] = end_x
            next_seg["y"] = end_y
            next_seg["hdg"] = end_hdg

        contiguous = (
            abs(next_seg["s"] - expected_s) <= 1e-8
            and math.hypot(next_seg["x"] - end_x, next_seg["y"] - end_y) <= adaptive_position_tol
            and abs(_normalize_angle(next_seg["hdg"] - end_hdg)) <= adaptive_heading_tol
        )

        if (
            same_curvature
            and contiguous
            and (
                merge_threshold is None
                or (current["length"] + next_seg["length"] <= merge_threshold)
            )
        ):
            current["length"] += next_seg["length"]
            continue

        merged.append(current)
        current = next_seg

    merged.append(current)
    return merged


def build_shoulder_profile(
    df_shoulder: Optional[DataFrame],
    *,
    offset_mapper: Optional[Callable[[float], float]] = None,
) -> List[Dict[str, float]]:
    if df_shoulder is None or len(df_shoulder) == 0:
        return []

    start_col = _find_column(df_shoulder, "offset", exclude=("end",))
    end_col = _find_column(df_shoulder, "end", "offset")
    left_col = _find_column(df_shoulder, "左", "路肩") or _find_column(df_shoulder, "left")
    right_col = _find_column(df_shoulder, "右", "路肩") or _find_column(df_shoulder, "right")
    path_col = _find_column(df_shoulder, "path")
    retrans_col = _find_column(df_shoulder, "is", "retransmission")

    if start_col is None or end_col is None:
        return []

    best_path = _select_best_path(df_shoulder, path_col)

    entries: List[Tuple[float, float, Optional[float], Optional[float]]] = []
    origin_cm: Optional[float] = None

    for idx in range(len(df_shoulder)):
        row = df_shoulder.iloc[idx]

        if retrans_col is not None:
            retrans_val = str(row[retrans_col]).strip().lower()
            if retrans_val == "true":
                continue

        if best_path is not None and row[path_col] != best_path:
            continue

        start_cm = _to_float(row[start_col])
        end_cm = _to_float(row[end_col])
        if start_cm is None or end_cm is None:
            continue

        if origin_cm is None or start_cm < origin_cm:
            origin_cm = start_cm

        left_val = _to_float(row[left_col]) if left_col is not None else None
        right_val = _to_float(row[right_col]) if right_col is not None else None

        entries.append((start_cm, end_cm, left_val, right_val))

    if origin_cm is None:
        return []

    grouped: Dict[Tuple[float, float], Dict[str, List[float]]] = {}

    for start_cm, end_cm, left_val, right_val in entries:
        start_m = max(0.0, (start_cm - origin_cm) * 0.01)
        end_m = max(0.0, (end_cm - origin_cm) * 0.01)
        if end_m <= start_m:
            continue

        entry = grouped.setdefault(_prepare_segment_key(start_m, end_m), {"left": [], "right": []})

        if left_val is not None:
            entry["left"].append(left_val * 0.01)

        if right_val is not None:
            entry["right"].append(right_val * 0.01)

    profile: List[Dict[str, float]] = []
    for (start_m, end_m), values in sorted(grouped.items(), key=lambda item: item[0]):
        s0 = float(offset_mapper(start_m)) if offset_mapper is not None else float(start_m)
        s1 = float(offset_mapper(end_m)) if offset_mapper is not None else float(end_m)
        if s1 <= s0:
            continue

        left_width = sum(values["left"]) / len(values["left"]) if values["left"] else 0.0
        right_width = sum(values["right"]) / len(values["right"]) if values["right"] else 0.0

        profile.append({"s0": s0, "s1": s1, "left": left_width, "right": right_width})

    return profile
