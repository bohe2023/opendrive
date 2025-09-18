import pandas as pd

def read_csv_any(path, encodings=("utf-8", "cp932", "gb18030")) -> pd.DataFrame:
    """
    Try multiple encodings to read CSV robustly.
    """
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"Could not read CSV: {path}")
