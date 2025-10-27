from csv2xodr.simpletable import Series, notna

def mark_type_from_division_row(row: Series) -> str:
    """車線区分テーブルの情報から破線か実線かを素朴に推定する。"""
    for c in row.index:
        if ("破線" in c or "ペイント" in c) and notna(row[c]):
            try:
                if int(row[c]) == 1:
                    return "broken"
            except Exception:
                pass

    for c in row.index:
        lowered = str(c).strip().lower()
        if "lane line type" in lowered and notna(row[c]):
            try:
                v = int(float(str(row[c]).strip()))
            except Exception:
                continue
            if v in (2, 3):
                return "broken"
            return "solid"

    for c in row.index:
        if "種別" in c and notna(row[c]):
            try:
                v = int(row[c])
                return "solid" if v in (1, 2) else "broken"
            except Exception:
                pass
    return "solid"
