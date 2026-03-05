import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["avg_rating", "avg_review_count",
            "business_count", "median_income", "employment_rate"]


def add_feasibility_score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty dataframe in add_feasibility_score().")

    out = df.copy()
    scaler = MinMaxScaler()
    out[FEATURES] = scaler.fit_transform(out[FEATURES])

    out["feasibility_score"] = (
        0.25 * out["avg_rating"] +
        0.20 * out["avg_review_count"] +
        0.20 * out["business_count"] +
        0.20 * out["median_income"] +
        0.15 * out["employment_rate"]
    )
    return out


def add_class_label(df: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
    out = df.copy()
    out["class_label"] = (out["feasibility_score"] >= threshold).astype(int)
    return out
