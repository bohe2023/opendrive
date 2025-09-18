import os
from .utils import read_csv_any

def load_all(input_dir: str, cfg: dict) -> dict:
    files = cfg["files"]
    result = {}
    for key, name in files.items():
        p = os.path.join(input_dir, name)
        if os.path.exists(p):
            result[key] = read_csv_any(p, tuple(cfg.get("encoding_priority", ["utf-8", "cp932"])))
        else:
            result[key] = None
    return result
