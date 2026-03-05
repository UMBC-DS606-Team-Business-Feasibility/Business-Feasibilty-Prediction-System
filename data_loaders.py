import json
import pandas as pd


def load_yelp_business_ndjson(path: str, limit_rows: int | None = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows.append(json.loads(line))
            if limit_rows and (i + 1) >= limit_rows:
                break
    return pd.DataFrame(rows)


def load_census_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_bls_xlsx(path: str) -> pd.DataFrame:
    return pd.read_excel(path)
