import pandas as pd

def _offset_series(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return None, None
    off_cols = [c for c in df.columns if "Offset" in c]             # Offset[cm]
    end_cols = [c for c in df.columns if "End Offset" in c]         # End Offset[cm]
    off = df[off_cols[0]].astype(float) / 100.0 if off_cols else None
    end = df[end_cols[0]].astype(float) / 100.0 if end_cols else None
    return off, end

def make_sections(centerline: pd.DataFrame,
                  lane_link_df: pd.DataFrame = None,
                  lane_div_df: pd.DataFrame = None,
                  min_len: float = 0.01):
    """
    Collect split points from lane_link/lane_div offsets (in meters),
    produce lane sections [s0, s1).
    """
    splits = set([0.0, float(centerline["s"].iloc[-1])])

    for df in (lane_link_df, lane_div_df):
        off, end = _offset_series(df)
        if off is not None:
            for v in off.values:
                splits.add(float(v))
        if end is not None:
            for v in end.values:
                splits.add(float(v))

    splits = sorted(splits)
    sections = []
    for i in range(len(splits) - 1):
        s0, s1 = splits[i], splits[i + 1]
        if s1 - s0 > min_len:
            sections.append({"s0": s0, "s1": s1})
    return sections

def build_lane_topology(lane_link_df: pd.DataFrame):
    """
    Return:
      - lane topology hints dict (lanes_guess: list[int], lane_id_col: str)
      - unique lane IDs list (for future fine mapping)
    Rules:
      * If "Lane Number" column exists, we guess left/right counts.
      * We DO NOT include center(0) in lanes_guess; writer will create center.
    """
    if lane_link_df is None or len(lane_link_df) == 0:
        return {"lanes_guess": [1, -1], "lane_id_col": None}, []

    cols = list(lane_link_df.columns)
    lane_id_col = [c for c in cols if "Lane ID" in c or c.lower().strip() in ("lane_id", "lane id")]
    lane_id_col = lane_id_col[0] if lane_id_col else cols[-1]

    lane_num_col = [c for c in cols if "Lane Number" in c or c.lower().startswith("lane number")]
    lane_num_col = lane_num_col[0] if lane_num_col else None

    lanes = [1, -1]  # fallback 2 lanes (left=positive, right=negative)
    if lane_num_col:
        try:
            n = int(lane_link_df[lane_num_col].iloc[0])
            # guess: n means total including center → left=n//2, right=n - left - 1(center)
            left = max(1, n // 2)
            right = max(1, n - left - 1)
            left_ids = list(range(1, left + 1))
            right_ids = list(range(-1, -right - 1, -1))
            lanes = left_ids + right_ids  # no center(0)
        except Exception:
            pass

    unique_ids = pd.unique(lane_link_df[lane_id_col]).tolist() if lane_id_col else []
    return {"lanes_guess": lanes, "lane_id_col": lane_id_col}, unique_ids
