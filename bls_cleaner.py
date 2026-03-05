import pandas as pd
import numpy as np


def bls_laus_metro_to_city_employment(bls: pd.DataFrame, target_year: int = 2024) -> pd.DataFrame:
    df = bls.copy().dropna(how="all")

    def row_has_year(row) -> bool:
        for v in row:
            if pd.isna(v):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) and int(v) == target_year:
                return True
            s = str(v).strip()
            if s in {str(target_year), f"{target_year}.0"}:
                return True
        return False

    header_row = None
    for i in df.index:
        if row_has_year(df.loc[i].values):
            header_row = i
            break
    if header_row is None:
        raise ValueError(
            f"Could not find header row containing year {target_year} in BLS sheet.")

    header = [str(x).strip() if not pd.isna(
        x) else "" for x in df.loc[header_row].tolist()]
    df2 = df.loc[header_row + 1:].copy()
    df2.columns = header
    df2 = df2.loc[:, [c for c in df2.columns if c != ""]]

    # year column
    year_col = None
    for c in df2.columns:
        if str(c).strip() in {str(target_year), f"{target_year}.0"}:
            year_col = c
            break
    if year_col is None:
        for c in df2.columns:
            try:
                if int(float(str(c))) == target_year:
                    year_col = c
                    break
            except:
                pass
    if year_col is None:
        raise ValueError(
            f"Found header but not {target_year} column. Columns: {list(df2.columns)}")

    # detect area name column (avoid numeric codes)
    best_col, best_score = None, -1
    for c in df2.columns:
        if c == year_col:
            continue
        sample = df2[c].dropna().astype(str).head(200)
        if sample.empty:
            continue
        has_state = sample.str.contains(r",\s*[A-Z]{2}\b").mean()
        has_hyphen = sample.str.contains(r"-").mean()
        mostly_digits = sample.str.fullmatch(r"\d+").mean()
        score = (2.0 * has_state) + (1.0 * has_hyphen) - (3.0 * mostly_digits)
        if score > best_score:
            best_score = score
            best_col = c

    if best_col is None:
        raise ValueError(
            "Could not detect metro area name column in BLS sheet.")

    out = df2[[best_col, year_col]].copy()
    out = out.rename(columns={best_col: "area", year_col: "unemployment_rate"})
    out["unemployment_rate"] = pd.to_numeric(
        out["unemployment_rate"], errors="coerce")
    out = out.dropna(subset=["unemployment_rate"])

    out["employment_rate"] = 100.0 - out["unemployment_rate"]

    out["city"] = (
        out["area"].astype(str)
        .str.split(",").str[0]
        .str.split("-").str[0]
        .str.strip()
        .str.upper()
    )
    return out[["city", "employment_rate"]]
