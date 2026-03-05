import pandas as pd


def build_master_city_table(yelp_city: pd.DataFrame, census_city: pd.DataFrame, bls_city: pd.DataFrame) -> pd.DataFrame:
    df = yelp_city.merge(census_city, on="city", how="inner")
    df = df.merge(bls_city, on="city", how="inner")
    if df.empty:
        raise ValueError(
            "Master dataset is empty after merge. Check city overlap across sources.")
    return df
