import pandas as pd
import re


def _norm_city(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[A-Z]{2}\s+", "", s)  # drop "AB Edmonton"
    return s


def clean_yelp_business(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["business_id", "city", "state", "stars", "review_count"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()

    df = df.dropna(subset=["business_id", "city", "stars", "review_count"])
    df["city"] = df["city"].apply(_norm_city)
    df["state"] = df["state"].astype(str).str.strip().str.upper()

    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    df = df.dropna(subset=["stars", "review_count"])
    return df


def city_level_yelp_features(yelp: pd.DataFrame) -> pd.DataFrame:
    out = (
        yelp.groupby("city")
        .agg(
            avg_rating=("stars", "mean"),
            avg_review_count=("review_count", "mean"),
            business_count=("business_id", "count"),
        )
        .reset_index()
    )
    return out
