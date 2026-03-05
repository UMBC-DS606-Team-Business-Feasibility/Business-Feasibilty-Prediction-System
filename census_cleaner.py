import pandas as pd
import re


def census_b19013_to_city_income(census: pd.DataFrame):

    # Find columns that contain Estimate values
    income_cols = [
        c for c in census.columns
        if "Estimate" in str(c)
    ]

    if len(income_cols) == 0:
        raise ValueError(
            "Could not detect income estimate columns in census file.")

    # Find the row that actually contains numeric income values
    income_row_index = None

    for i in range(len(census)):
        row = census.iloc[i]
        numeric_count = 0

        for col in income_cols:
            val = str(row[col]).replace(",", "")
            if val.isdigit():
                numeric_count += 1

        if numeric_count > 3:   # enough numeric values → this is the data row
            income_row_index = i
            break

    if income_row_index is None:
        raise ValueError("Could not find the row containing income values.")

    income_row = census.iloc[income_row_index]

    rows = []

    for col in income_cols:

        city_full = col.split("!!")[0]

        val = income_row[col]

        if pd.isna(val):
            continue

        val = str(val).replace(",", "")

        match = re.search(r"\d+", val)

        if not match:
            continue

        city = city_full.split(",")[0]
        city = city.replace(" city", "")
        city = city.replace(" CDP", "")
        city = city.strip().upper()

        rows.append({
            "city": city,
            "median_income": float(match.group())
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No cities parsed from census file.")

    return df
