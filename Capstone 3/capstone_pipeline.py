from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PRESENTATION_DIR = BASE_DIR / "presentation"
METRO_DATA_DIR = BASE_DIR / "metro_data"

MPLCONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_TITLE = "Metropolitan Business Feasibility Prediction System"
PROJECT_SUBTITLE = (
    "Ranking and Predicting High-Opportunity U.S. Metropolitan Areas from "
    "County Business Patterns, ACS, and BLS data"
)
METRO_SAMPLE_LIMIT = (
    int(os.environ["METRO_SAMPLE_LIMIT"])
    if os.environ.get("METRO_SAMPLE_LIMIT")
    else None
)
RANDOM_STATE = 42
TARGET_QUANTILE = 0.70

# Auto-select newest CBP file available (prefer 2023, fall back to 2022)
def _resolve_cbp_path() -> tuple[Path, str]:
    """Return (zip_path, year_label) for the newest available CBP MSA file."""
    for year_short in ("23", "22"):
        candidate = METRO_DATA_DIR / f"cbp{year_short}msa.zip"
        if candidate.exists():
            return candidate, f"20{year_short}"
    raise FileNotFoundError(
        "No CBP MSA file found in metro_data/. Expected cbp23msa.zip or cbp22msa.zip."
    )

CBP_ZIP_PATH, CBP_YEAR = _resolve_cbp_path()
ACS_JSON_PATH = METRO_DATA_DIR / "acs_metro_income_population_2024.json"
BLS_XLSX_PATH = BASE_DIR / "bls_unemployment.xlsx"
BEA_RPP_PATH  = METRO_DATA_DIR / "bea_rpp_msa_2008_2024.csv"
GAZETTEER_PATH = METRO_DATA_DIR / "cbsa_gazetteer_2024.txt"

REGION_MAP = {
    "Northeast": {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"},
    "Midwest": {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"},
    "South": {
        "DE",
        "FL",
        "GA",
        "MD",
        "NC",
        "SC",
        "VA",
        "DC",
        "WV",
        "AL",
        "KY",
        "MS",
        "TN",
        "AR",
        "LA",
        "OK",
        "TX",
    },
    "West": {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"},
}

NAICS_LABELS = {
    "11": "Agriculture",
    "21": "Mining & Oil/Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "48": "Transportation & Warehousing",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Management of Companies",
    "56": "Administrative Support",
    "61": "Educational Services",
    "62": "Health Care",
    "71": "Arts & Recreation",
    "72": "Accommodation & Food",
    "81": "Other Services",
    "99": "Unclassified",
}

BUSINESS_CATEGORIES = {
    "restaurant": {
        "label": "Restaurant / Food Service",
        "sector_code": "72",
        "color": "#d62828",
    },
    "retail": {
        "label": "Retail Store",
        "sector_code": "44",
        "color": "#1d3557",
    },
    "clinic": {
        "label": "Health Clinic / Care Service",
        "sector_code": "62",
        "color": "#2a9d8f",
    },
    "professional_services": {
        "label": "Professional Services Firm",
        "sector_code": "54",
        "color": "#6d597a",
    },
    "personal_services": {
        "label": "Salon / Repair / Personal Services",
        "sector_code": "81",
        "color": "#f4a261",
    },
    "technology": {
        "label": "Technology / IT Services",
        "sector_code": "51",
        "color": "#023e8a",
    },
    "education": {
        "label": "Education / Childcare Services",
        "sector_code": "61",
        "color": "#e9c46a",
    },
    "construction": {
        "label": "Construction / Trades",
        "sector_code": "23",
        "color": "#264653",
    },
    "finance": {
        "label": "Finance / Insurance Services",
        "sector_code": "52",
        "color": "#8ecae6",
    },
}

# Tax Foundation State Business Tax Climate Index 2024 ranks (lower = better climate)
STATE_BUSINESS_CLIMATE: Dict[str, int] = {
    "WY": 1, "SD": 2, "AK": 3, "FL": 4, "MT": 5, "NH": 6, "NV": 7,
    "UT": 8, "IN": 9, "NC": 10, "TX": 11, "CO": 12, "TN": 13, "AZ": 14,
    "WI": 15, "OR": 16, "MO": 17, "NE": 18, "OK": 19, "ID": 20,
    "AR": 21, "GA": 22, "AL": 23, "MS": 24, "ND": 25, "SC": 26,
    "VA": 27, "HI": 28, "KS": 29, "MI": 30, "OH": 31, "IA": 32,
    "WA": 33, "ME": 34, "MN": 35, "KY": 36, "PA": 37, "MA": 38,
    "LA": 39, "IL": 40, "WV": 41, "DE": 42, "MD": 43, "VT": 44,
    "RI": 45, "CT": 46, "NJ": 47, "NY": 48, "CA": 49, "DC": 30,
}

STRATEGY_GUIDANCE = {
    "Established Leader": "Best for immediate entry into a proven, already-validated market.",
    "White-Space Opportunity": "Strong overall market with room for category expansion.",
    "Momentum Market": "Healthy metro fundamentals with improving labor-market momentum.",
    "Niche Cluster Bet": "Specialized category strength exists, but broader market support is thinner.",
    "Selective Expansion Bet": "Worth evaluating carefully, but not a top-priority entry market.",
}

MODEL_FEATURES = [
    "population",
    "emp",
    "est",
    "small_est_share",
    "mid_est_share",
    "large_est_share",
    "avg_emp_per_est",
    "annual_payroll_per_employee",
    "payroll_per_capita",
    "employment_per_100",
    "unemployment_trend_slope",
    "state_business_climate_norm",
    "cost_of_living_index",
    "housing_cost_index",
    "emp_share_11",
    "emp_share_21",
    "emp_share_22",
    "emp_share_23",
    "emp_share_31",
    "emp_share_42",
    "emp_share_44",
    "emp_share_48",
    "emp_share_51",
    "emp_share_52",
    "emp_share_53",
    "emp_share_54",
    "emp_share_55",
    "emp_share_56",
    "emp_share_61",
    "emp_share_62",
    "emp_share_71",
    "emp_share_72",
    "emp_share_81",
    "emp_share_99",
]


def scale_01(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - min_value) / (max_value - min_value)


def region_for_state(state_abbr: str) -> str:
    for region, states in REGION_MAP.items():
        if state_abbr in states:
            return region
    return "Other"


def metro_sample_scope(selected_count: int, overlap_count: int) -> str:
    if selected_count >= overlap_count:
        return f"all {overlap_count} overlapping metros"
    return f"top {selected_count} metros by population"


def prettify_feature_name(name: str) -> str:
    pretty = {
        "population": "Population",
        "emp": "Employment",
        "est": "Establishments",
        "small_est_share": "Share of Small Establishments",
        "mid_est_share": "Share of Mid-size Establishments",
        "large_est_share": "Share of Large Establishments",
        "avg_emp_per_est": "Average Employees per Establishment",
        "annual_payroll_per_employee": "Annual Payroll per Employee",
        "payroll_per_capita": "Payroll per Capita",
        "employment_per_100": "Employment per 100 Residents",
        "unemployment_trend_slope": "Unemployment Trend Slope (2020-2024)",
        "state_business_climate_norm": "State Business Climate Score",
        "cost_of_living_index": "Cost of Living Index (BEA RPP, 100 = US avg)",
        "housing_cost_index": "Housing Cost Index (BEA RPP, 100 = US avg)",
        "real_median_income": "Real Median Income (PPP-adjusted)",
        "recent_unemployment_change": "Recent Unemployment Change (2024 - 2023)",
    }
    if name in pretty:
        return pretty[name]
    if name.startswith("emp_share_"):
        code = name.split("_")[-1]
        return f"Employment Share: {NAICS_LABELS.get(code, code)}"
    return name.replace("_", " ").title()


def normalize_metro_title(title: str) -> str:
    value = str(title).upper().strip()
    value = value.replace(" METRO AREA", "")
    value = value.replace(" ST. ", " SAINT ")
    value = value.replace("–", "-")
    value = re.sub(r"\s+", " ", value)
    return value


def read_xlsx_sheet(path: Path) -> pd.DataFrame:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path) as workbook:
        shared_strings: List[str] = []
        shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        for item in shared_root.findall(f"{{{namespace}}}si"):
            shared_strings.append("".join((node.text or "") for node in item.iter(f"{{{namespace}}}t")))

        sheet_root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows: List[List[object]] = []
        max_col = 0

        def col_index(ref: str) -> int:
            letters = "".join(ch for ch in ref if ch.isalpha())
            index = 0
            for ch in letters:
                index = index * 26 + (ord(ch.upper()) - 64)
            return index

        for row in sheet_root.findall(f".//{{{namespace}}}row"):
            values: Dict[int, object] = {}
            for cell in row.findall(f"{{{namespace}}}c"):
                index = col_index(cell.attrib.get("r", ""))
                max_col = max(max_col, index)
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{namespace}}}v")
                value = value_node.text if value_node is not None else None
                if value is not None and cell_type == "s":
                    value = shared_strings[int(value)]
                values[index] = value
            if values:
                rows.append([values.get(i) for i in range(1, max_col + 1)])

    return pd.DataFrame(rows)


def load_cbp_raw(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        # Find the .txt member regardless of year (cbp22msa.txt or cbp23msa.txt)
        txt_name = next((n for n in archive.namelist() if n.lower().endswith(".txt")), None)
        if txt_name is None:
            raise RuntimeError(f"No .txt file found inside {path.name}")
        with archive.open(txt_name) as handle:
            cbp = pd.read_csv(handle, dtype={"msa": str, "naics": str})
    return cbp


def load_metro_centroids(path: Path) -> pd.DataFrame:
    """Load Census Bureau CBSA gazetteer (lat/lon centroids per metro).

    Returns a DataFrame with columns: cbsa, lat, lon. Missing file → empty frame.
    """
    if not path.exists():
        return pd.DataFrame(columns=["cbsa", "lat", "lon"])
    raw = pd.read_csv(path, sep="\t", dtype={"GEOID": str})
    raw.columns = [c.strip() for c in raw.columns]
    out = raw.rename(columns={
        "GEOID": "cbsa", "INTPTLAT": "lat", "INTPTLONG": "lon",
    })[["cbsa", "lat", "lon"]].copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def load_bea_rpp(path: Path, year: int = 2024) -> pd.DataFrame:
    """Load BEA Regional Price Parities for MSAs.

    Returns a DataFrame keyed by `cbsa` (str) with columns:
      - cost_of_living_index    (LineCode 1, "All items"; 100 = US average)
      - housing_cost_index      (LineCode 3, "Services: Housing")

    If the file is missing, returns an empty frame so the caller can fall back.
    """
    if not path.exists():
        return pd.DataFrame(columns=["cbsa", "cost_of_living_index", "housing_cost_index"])

    raw = pd.read_csv(path, dtype={"GeoFIPS": str, "LineCode": str})
    raw["GeoFIPS"] = raw["GeoFIPS"].astype(str).str.strip().str.replace('"', "", regex=False)
    # Pick the most recent year column actually present
    year_col = str(year)
    if year_col not in raw.columns:
        numeric_year_cols = [c for c in raw.columns if c.isdigit()]
        year_col = max(numeric_year_cols, key=int) if numeric_year_cols else None
    if year_col is None:
        return pd.DataFrame(columns=["cbsa", "cost_of_living_index", "housing_cost_index"])

    raw[year_col] = pd.to_numeric(raw[year_col], errors="coerce")
    line1 = raw[raw["LineCode"] == "1"][["GeoFIPS", year_col]].rename(
        columns={"GeoFIPS": "cbsa", year_col: "cost_of_living_index"}
    )
    line3 = raw[raw["LineCode"] == "3"][["GeoFIPS", year_col]].rename(
        columns={"GeoFIPS": "cbsa", year_col: "housing_cost_index"}
    )
    rpp = line1.merge(line3, on="cbsa", how="outer")
    # 5-digit MSA CBSA codes only (drop "00000" US row, "00999" non-metro, state aggregates, etc.)
    rpp = rpp[rpp["cbsa"].str.match(r"^\d{5}$") & (rpp["cbsa"] != "00000")]
    return rpp.reset_index(drop=True)


def load_acs_metro(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    acs = pd.DataFrame(raw[1:], columns=raw[0]).rename(
        columns={
            "metropolitan statistical area/micropolitan statistical area": "cbsa",
            "B19013_001E": "median_income",
            "B01003_001E": "population",
        }
    )
    acs["cbsa"] = acs["cbsa"].astype(str)
    acs["median_income"] = pd.to_numeric(acs["median_income"], errors="coerce")
    acs["population"] = pd.to_numeric(acs["population"], errors="coerce")
    return acs


def load_bls_metro(path: Path) -> pd.DataFrame:
    raw = read_xlsx_sheet(path)
    header = raw.iloc[1].tolist()
    body = raw.iloc[2:].copy()
    body.columns = header

    bls = body[
        ["Metropolitan area title", "2020", "2021", "2022", "2023", "2024"]
    ].copy()
    bls.columns = [
        "bls_title",
        "unemployment_2020",
        "unemployment_2021",
        "unemployment_2022",
        "unemployment_2023",
        "unemployment_2024",
    ]
    for column in [
        "unemployment_2020",
        "unemployment_2021",
        "unemployment_2022",
        "unemployment_2023",
        "unemployment_2024",
    ]:
        bls[column] = pd.to_numeric(bls[column], errors="coerce")
    bls["unemployment_rate"] = bls["unemployment_2024"]
    bls["recent_unemployment_change"] = (
        bls["unemployment_2024"] - bls["unemployment_2023"]
    )
    bls = bls.dropna(subset=["unemployment_rate"]).reset_index(drop=True)
    bls["norm_title"] = bls["bls_title"].map(normalize_metro_title)
    return bls


def build_selected_metros(cbp: pd.DataFrame, acs: pd.DataFrame, bls: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    acs_metro = acs[acs["NAME"].str.endswith("Metro Area")].copy()
    acs_metro["metro_title"] = acs_metro["NAME"].str.replace(" Metro Area$", "", regex=True)
    acs_metro["norm_title"] = acs_metro["metro_title"].map(normalize_metro_title)

    # BLS uses "St. Louis, MO-IL1" in the workbook.
    acs_metro.loc[
        acs_metro["metro_title"] == "St. Louis, MO-IL",
        "norm_title",
    ] = "SAINT LOUIS, MO-IL1"

    merged = acs_metro.merge(
        bls[
            [
                "norm_title",
                "bls_title",
                "unemployment_rate",
                "recent_unemployment_change",
                "unemployment_2020",
                "unemployment_2021",
                "unemployment_2022",
                "unemployment_2023",
            ]
        ],
        on="norm_title",
        how="inner",
    )
    merged = merged[merged["cbsa"].isin(set(cbp["msa"].unique()))].copy()
    overlap_count = len(merged)

    merged = merged.sort_values("population", ascending=False).copy()
    if METRO_SAMPLE_LIMIT is None:
        selected = merged.copy()
    else:
        selected = merged.head(min(METRO_SAMPLE_LIMIT, overlap_count)).copy()
    selected["primary_state"] = selected["NAME"].str.extract(r",\s*([A-Z]{2})(?:-[A-Z]{2})*\s+Metro Area$")
    selected["region"] = selected["primary_state"].map(region_for_state)
    return selected.reset_index(drop=True), overlap_count


def engineer_cbp_features(cbp: pd.DataFrame, selected_cbsa: Iterable[str]) -> tuple[pd.DataFrame, int]:
    filtered = cbp[cbp["msa"].isin(set(selected_cbsa))].copy()
    raw_row_count = len(filtered)

    numeric_cols = [
        "emp",
        "qp1",
        "ap",
        "est",
        "n<5",
        "n5_9",
        "n10_19",
        "n20_49",
        "n50_99",
        "n100_249",
        "n250_499",
        "n500_999",
        "n1000",
    ]
    for column in numeric_cols:
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce").fillna(0)

    total_rows = filtered[filtered["naics"] == "------"].copy()
    total_rows["small_est_share"] = (
        total_rows["n<5"] + total_rows["n5_9"] + total_rows["n10_19"] + total_rows["n20_49"]
    ) / total_rows["est"]
    total_rows["mid_est_share"] = (total_rows["n50_99"] + total_rows["n100_249"]) / total_rows["est"]
    total_rows["large_est_share"] = (
        total_rows["n250_499"] + total_rows["n500_999"] + total_rows["n1000"]
    ) / total_rows["est"]
    total_rows["avg_emp_per_est"] = total_rows["emp"] / total_rows["est"]
    total_rows["annual_payroll_per_employee"] = total_rows["ap"] / total_rows["emp"].replace(0, np.nan)
    total_rows["annual_payroll_per_est"] = total_rows["ap"] / total_rows["est"].replace(0, np.nan)

    sector_rows = filtered[filtered["naics"].str.match(r"^\d{2}----$")].copy()
    sector_rows["sector_code"] = sector_rows["naics"].str[:2]

    sector_emp = sector_rows.pivot_table(
        index="msa",
        columns="sector_code",
        values="emp",
        aggfunc="sum",
        fill_value=0,
    )
    sector_emp_share = sector_emp.div(sector_emp.sum(axis=1), axis=0).fillna(0)

    for code in NAICS_LABELS:
        if code not in sector_emp_share.columns:
            sector_emp_share[code] = 0.0
    sector_emp_share = sector_emp_share[sorted(sector_emp_share.columns)]
    sector_emp_share.columns = [f"emp_share_{code}" for code in sector_emp_share.columns]

    sector_est = sector_rows.pivot_table(
        index="msa",
        columns="sector_code",
        values="est",
        aggfunc="sum",
        fill_value=0,
    )
    for code in NAICS_LABELS:
        if code not in sector_est.columns:
            sector_est[code] = 0.0
    sector_est = sector_est[sorted(sector_est.columns)]
    sector_est_share = sector_est.div(sector_est.sum(axis=1), axis=0).fillna(0)
    sector_est_counts = sector_est.copy()

    sector_est_counts.columns = [f"sector_est_{code}" for code in sector_est_counts.columns]
    sector_est_share.columns = [f"est_share_{code}" for code in sector_est_share.columns]

    entropy = -(
        sector_est_share.replace(0, np.nan) * np.log(sector_est_share.replace(0, np.nan))
    ).sum(axis=1).fillna(0)

    features = total_rows[
        [
            "msa",
            "emp",
            "qp1",
            "ap",
            "est",
            "small_est_share",
            "mid_est_share",
            "large_est_share",
            "avg_emp_per_est",
            "annual_payroll_per_employee",
            "annual_payroll_per_est",
        ]
    ].copy()
    features = features.merge(sector_emp_share, left_on="msa", right_index=True, how="left")
    features = features.merge(sector_est_counts, left_on="msa", right_index=True, how="left")
    features = features.merge(sector_est_share, left_on="msa", right_index=True, how="left")
    features["industry_entropy"] = features["msa"].map(entropy)
    return features, raw_row_count


def build_model_dataset(
    selected_metros: pd.DataFrame,
    cbp_features: pd.DataFrame,
    bea_rpp: pd.DataFrame | None = None,
    centroids: pd.DataFrame | None = None,
) -> pd.DataFrame:
    dataset = selected_metros.merge(cbp_features, left_on="cbsa", right_on="msa", how="inner")

    # Merge BEA Regional Price Parities (cost of living + housing index, 100 = US avg).
    # Missing values filled with 100 (national average) so metros without RPP coverage are neutral.
    if bea_rpp is not None and not bea_rpp.empty:
        dataset = dataset.merge(bea_rpp, on="cbsa", how="left")
    if "cost_of_living_index" not in dataset.columns:
        dataset["cost_of_living_index"] = 100.0
    if "housing_cost_index" not in dataset.columns:
        dataset["housing_cost_index"] = 100.0
    dataset["cost_of_living_index"] = dataset["cost_of_living_index"].fillna(100.0)
    dataset["housing_cost_index"]   = dataset["housing_cost_index"].fillna(100.0)

    # Merge Census CBSA gazetteer for lat/lon (used by bubble map). Missing → leave blank.
    if centroids is not None and not centroids.empty:
        dataset = dataset.merge(centroids, on="cbsa", how="left")

    # Real (purchasing-power-adjusted) median income — what your dollars actually buy.
    dataset["real_median_income"] = (
        dataset["median_income"] * 100.0 / dataset["cost_of_living_index"]
    )

    dataset["establishments_per_10k"] = dataset["est"] / dataset["population"] * 10000
    dataset["employment_per_100"] = dataset["emp"] / dataset["population"] * 100
    dataset["payroll_per_capita"] = dataset["ap"] / dataset["population"]

    for code in NAICS_LABELS:
        count_col = f"sector_est_{code}"
        if count_col in dataset.columns:
            dataset[f"sector_est_per_10k_{code}"] = dataset[count_col] / dataset["population"] * 10000

    # Unemployment trend slope: linear slope over 2020-2024 (negative = improving)
    unemp_years = np.array([2020.0, 2021.0, 2022.0, 2023.0, 2024.0])
    unemp_years_c = unemp_years - unemp_years.mean()

    def _trend_slope(row: pd.Series) -> float:
        vals = np.array([
            row.get("unemployment_2020", np.nan),
            row.get("unemployment_2021", np.nan),
            row.get("unemployment_2022", np.nan),
            row.get("unemployment_2023", np.nan),
            float(row.get("unemployment_rate", np.nan)),
        ], dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            return 0.0
        return float(np.polyfit(unemp_years_c[mask], vals[mask], 1)[0])

    dataset["unemployment_trend_slope"] = dataset.apply(_trend_slope, axis=1)

    # State business climate: invert rank so higher = friendlier business environment
    dataset["state_business_climate_rank"] = (
        dataset["primary_state"].map(STATE_BUSINESS_CLIMATE).fillna(25).astype(float)
    )
    dataset["state_business_climate_norm"] = scale_01(
        dataset["state_business_climate_rank"].max() - dataset["state_business_climate_rank"]
    )

    # Sector saturation: est_per_10k relative to national median (1.0 = at median, >1 = denser)
    for biz_key, meta in BUSINESS_CATEGORIES.items():
        code = meta["sector_code"]
        dens_col = f"sector_est_per_10k_{code}"
        if dens_col in dataset.columns:
            nat_median = float(dataset[dens_col].median())
            dataset[f"saturation_{code}"] = (
                dataset[dens_col].div(nat_median if nat_median > 0 else 1.0).clip(0, 5)
            )

    dataset["median_income_norm"]      = scale_01(dataset["median_income"])
    dataset["real_median_income_norm"] = scale_01(dataset["real_median_income"])
    dataset["inverse_unemployment_norm"] = scale_01(dataset["unemployment_rate"].max() - dataset["unemployment_rate"])
    dataset["annual_payroll_per_employee_norm"] = scale_01(dataset["annual_payroll_per_employee"])
    dataset["establishments_per_10k_norm"] = scale_01(dataset["establishments_per_10k"])
    dataset["industry_entropy_norm"]   = scale_01(dataset["industry_entropy"])
    # Cost-of-living penalty (higher cost = lower score). Inverted & scaled to [0,1].
    dataset["affordability_norm"] = scale_01(
        dataset["cost_of_living_index"].max() - dataset["cost_of_living_index"]
    )

    # Updated opportunity-score weights — now incorporating cost of living.
    # Real income (PPP-adjusted) replaces gross income; housing affordability is a small dampener.
    dataset["opportunity_score"] = (
        0.25 * dataset["real_median_income_norm"]      # was median_income_norm @ 25%
        + 0.20 * dataset["inverse_unemployment_norm"]   # was 25%
        + 0.15 * dataset["annual_payroll_per_employee_norm"]  # was 20%
        + 0.15 * dataset["establishments_per_10k_norm"]
        + 0.15 * dataset["industry_entropy_norm"]
        + 0.10 * dataset["affordability_norm"]          # NEW — penalizes HCOL metros
    )
    dataset["high_opportunity"] = (
        dataset["opportunity_score"] >= dataset["opportunity_score"].quantile(TARGET_QUANTILE)
    ).astype(int)
    return dataset


def validation_frame(records: List[Dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values("validation_score", ascending=False).reset_index(drop=True)


def cluster_metros(dataset: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """K-Means clustering of metros; returns (cluster_frame, cluster_profiles, pca_variance)."""
    X = dataset[MODEL_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(X_scaled)

    result = dataset[
        ["metro_title", "primary_state", "region", "opportunity_score",
         "median_income", "unemployment_rate", "population",
         "establishments_per_10k", "annual_payroll_per_employee", "industry_entropy"]
    ].copy()
    result["cluster"] = labels
    result["pca_1"] = pca_coords[:, 0]
    result["pca_2"] = pca_coords[:, 1]

    # Profile each cluster and assign a descriptive name
    profiling_cols = [
        "opportunity_score", "median_income", "unemployment_rate",
        "population", "establishments_per_10k", "annual_payroll_per_employee",
    ]
    profiles = result.groupby("cluster")[profiling_cols].mean()

    cluster_names: Dict[int, str] = {}
    used_names: set = set()
    for c in range(n_clusters):
        row = profiles.loc[c]
        candidates = []
        if row["opportunity_score"] >= profiles["opportunity_score"].quantile(0.75):
            candidates.append("Elite Opportunity Market")
        if row["median_income"] >= profiles["median_income"].quantile(0.75) and \
                row["unemployment_rate"] <= profiles["unemployment_rate"].quantile(0.35):
            candidates.append("Affluent Low-Risk Hub")
        if row["population"] >= profiles["population"].quantile(0.75):
            candidates.append("Major Metro Engine")
        if row["establishments_per_10k"] >= profiles["establishments_per_10k"].quantile(0.65):
            candidates.append("Business-Dense Market")
        if row["unemployment_rate"] >= profiles["unemployment_rate"].quantile(0.65):
            candidates.append("High-Unemployment Market")
        if row["annual_payroll_per_employee"] <= profiles["annual_payroll_per_employee"].quantile(0.35):
            candidates.append("Value / Cost-Efficient Market")
        name = next((n for n in candidates if n not in used_names), f"Mixed Market {c}")
        cluster_names[c] = name
        used_names.add(name)

    result["cluster_name"] = result["cluster"].map(cluster_names)
    profiles["cluster_name"] = pd.Series(cluster_names)
    return result, profiles, pca.explained_variance_ratio_


def compute_sensitivity(dataset: pd.DataFrame) -> pd.DataFrame:
    """Test 5 opportunity-score weight scenarios; return per-metro rank stability."""
    scenarios: Dict[str, Dict[str, float]] = {
        "Income-Heavy":      {"income": 0.40, "unemp": 0.20, "payroll": 0.15, "density": 0.15, "entropy": 0.10},
        "Employment-Heavy":  {"income": 0.15, "unemp": 0.40, "payroll": 0.20, "density": 0.15, "entropy": 0.10},
        "Balanced (default)":{"income": 0.25, "unemp": 0.25, "payroll": 0.20, "density": 0.15, "entropy": 0.15},
        "Density-Heavy":     {"income": 0.20, "unemp": 0.15, "payroll": 0.15, "density": 0.35, "entropy": 0.15},
        "Diversity-Heavy":   {"income": 0.20, "unemp": 0.15, "payroll": 0.15, "density": 0.15, "entropy": 0.35},
    }
    rank_frames: List[pd.Series] = []
    for scenario_name, w in scenarios.items():
        score = (
            w["income"]  * dataset["median_income_norm"]
            + w["unemp"]   * dataset["inverse_unemployment_norm"]
            + w["payroll"] * dataset["annual_payroll_per_employee_norm"]
            + w["density"] * dataset["establishments_per_10k_norm"]
            + w["entropy"] * dataset["industry_entropy_norm"]
        )
        rank = score.rank(method="min", ascending=False).astype(int)
        rank_frames.append(rank.rename(scenario_name))

    sens = pd.concat(rank_frames, axis=1)
    sens.index = dataset["metro_title"]
    sens["rank_mean"] = sens.mean(axis=1)
    sens["rank_std"]  = sens.std(axis=1)
    sens["stability"] = sens["rank_std"].apply(lambda s: "Stable" if s <= 15 else ("Moderate" if s <= 35 else "Volatile"))
    return sens.sort_values("rank_mean").reset_index()


def evaluate_models(dataset: pd.DataFrame) -> Dict[str, object]:
    X = dataset[MODEL_FEATURES].fillna(0)
    y_class = dataset["high_opportunity"]
    y_reg = dataset["unemployment_rate"]

    X_temp, X_test, y_class_temp, y_class_test, y_reg_temp, y_reg_test = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )
    X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
        X_temp,
        y_class_temp,
        y_reg_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_class_temp,
    )

    _lr = LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear")
    _rf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    _gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE,
    )
    classifier_candidates = {
        "LogisticRegression": _lr,
        "RandomForestClassifier": _rf,
        "GradientBoostingClassifier": _gbc,
        "VotingClassifier": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear")),
                ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                              random_state=RANDOM_STATE, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                  learning_rate=0.05, subsample=0.8,
                                                  min_samples_leaf=5, random_state=RANDOM_STATE)),
            ],
            voting="soft",
        ),
    }
    regressor_candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    preprocessor = ColumnTransformer([("num", StandardScaler(), MODEL_FEATURES)])

    classifier_validation: List[Dict[str, object]] = []
    best_classifier_name = ""
    best_classifier_model = None
    best_classifier_score = -1.0

    for name, model in classifier_candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_class_train)
        predictions = pipeline.predict(X_val)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        f1 = float(f1_score(y_class_val, predictions))
        classifier_validation.append(
            {
                "model": name,
                "validation_score": f1,
                "validation_accuracy": float(accuracy_score(y_class_val, predictions)),
                "validation_f1": f1,
                "validation_auc": float(roc_auc_score(y_class_val, probabilities)),
            }
        )
        if f1 > best_classifier_score:
            best_classifier_name = name
            best_classifier_model = model
            best_classifier_score = f1

    regressor_validation: List[Dict[str, object]] = []
    best_regressor_name = ""
    best_regressor_model = None
    best_regressor_score = float("inf")

    for name, model in regressor_candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_reg_train)
        predictions = pipeline.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_reg_val, predictions)))
        regressor_validation.append(
            {
                "model": name,
                "validation_score": -rmse,
                "validation_rmse": rmse,
                "validation_mae": float(mean_absolute_error(y_reg_val, predictions)),
                "validation_r2": float(r2_score(y_reg_val, predictions)),
            }
        )
        if rmse < best_regressor_score:
            best_regressor_name = name
            best_regressor_model = model
            best_regressor_score = rmse

    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_class_trainval = pd.concat([y_class_train, y_class_val], axis=0)
    y_reg_trainval = pd.concat([y_reg_train, y_reg_val], axis=0)

    # --- GridSearchCV tuning on best selected model ---
    tuning_grids: Dict[str, dict] = {
        "GradientBoostingClassifier": {
            "model__n_estimators": [200, 300],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.05, 0.1],
        },
        "LogisticRegression": {
            "model__C": [0.1, 0.5, 1.0, 5.0],
        },
        "RandomForestClassifier": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8],
        },
        "VotingClassifier": {},
    }
    grid_param = tuning_grids.get(best_classifier_name, {})
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_clf_pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_classifier_model)])
    if grid_param:
        gs = GridSearchCV(
            base_clf_pipeline, grid_param,
            cv=cv_strat, scoring="f1", n_jobs=-1, refit=True,
        )
        gs.fit(X_trainval, y_class_trainval)
        classifier_pipeline = gs.best_estimator_
        best_classifier_name = f"{best_classifier_name} (tuned)"
    else:
        classifier_pipeline = base_clf_pipeline
        classifier_pipeline.fit(X_trainval, y_class_trainval)

    class_test_pred = classifier_pipeline.predict(X_test)
    class_test_proba = classifier_pipeline.predict_proba(X_test)[:, 1]

    # --- CalibratedClassifierCV for well-calibrated market-strength probabilities ---
    calibrated_clf = CalibratedClassifierCV(best_classifier_model, cv=5, method="isotonic")
    calibrated_pipeline = Pipeline([("preprocessor", preprocessor), ("model", calibrated_clf)])
    calibrated_pipeline.fit(X_trainval, y_class_trainval)

    regressor_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", best_regressor_model)]
    )
    regressor_pipeline.fit(X_trainval, y_reg_trainval)
    reg_test_pred = regressor_pipeline.predict(X_test)

    class_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    class_cv_scores = cross_val_score(
        Pipeline([("preprocessor", preprocessor), ("model", best_classifier_model)]),
        X_trainval,
        y_class_trainval,
        cv=class_cv,
        scoring="f1",
    )
    reg_cv_scores = cross_val_score(
        Pipeline([("preprocessor", preprocessor), ("model", best_regressor_model)]),
        X_trainval,
        y_reg_trainval,
        cv=5,
        scoring="r2",
    )

    # Per-fold misclassification tracking + full assignments for map
    fold_misc_records: List[Dict[str, object]] = []
    fold_assignment_records: List[Dict[str, object]] = []
    trainval_indices = X_trainval.index.tolist()
    X_trainval_arr = X_trainval.reset_index(drop=True)
    y_trainval_arr = y_class_trainval.reset_index(drop=True)

    def _row_geo(orig_idx):
        r = dataset.loc[orig_idx]
        return {
            "metro_title": r["metro_title"],
            "primary_state": r["primary_state"],
            "region": r["region"],
            "lat": r.get("lat", np.nan),
            "lon": r.get("lon", np.nan),
            "opportunity_score": round(float(r["opportunity_score"]), 4),
        }

    for fold_num, (tr_idx, val_idx) in enumerate(
        class_cv.split(X_trainval_arr, y_trainval_arr), start=1
    ):
        fold_pipe = Pipeline([("preprocessor", preprocessor), ("model", best_classifier_model)])
        fold_pipe.fit(X_trainval_arr.iloc[tr_idx], y_trainval_arr.iloc[tr_idx])
        fold_preds = fold_pipe.predict(X_trainval_arr.iloc[val_idx])
        fold_proba = fold_pipe.predict_proba(X_trainval_arr.iloc[val_idx])[:, 1]

        # Record training metros
        for ti in tr_idx:
            orig_idx = trainval_indices[ti]
            base = _row_geo(orig_idx)
            base.update({
                "scenario": f"Fold {fold_num}",
                "split": "Training",
                "actual_label": "",
                "predicted_label": "",
                "predicted_probability_high": np.nan,
                "error_type": "",
                "is_misclassified": False,
            })
            fold_assignment_records.append(base)

        # Record validation metros (with predictions)
        for local_i, (pred, prob, actual) in enumerate(
            zip(fold_preds, fold_proba, y_trainval_arr.iloc[val_idx])
        ):
            orig_idx = trainval_indices[val_idx[local_i]]
            base = _row_geo(orig_idx)
            is_correct = (pred == actual)
            if is_correct:
                error = "Correct"
            else:
                error = "False Negative" if actual == 1 else "False Positive"
                fold_misc_records.append({
                    "fold": fold_num,
                    **{k: base[k] for k in ("metro_title", "primary_state", "region", "opportunity_score")},
                    "actual_label": "High" if actual == 1 else "Low/Medium",
                    "predicted_label": "High" if pred == 1 else "Low/Medium",
                    "predicted_probability_high": round(float(prob), 4),
                    "error_type": error,
                })
            base.update({
                "scenario": f"Fold {fold_num}",
                "split": "Validation",
                "actual_label": "High" if actual == 1 else "Low/Medium",
                "predicted_label": "High" if pred == 1 else "Low/Medium",
                "predicted_probability_high": round(float(prob), 4),
                "error_type": error,
                "is_misclassified": not is_correct,
            })
            fold_assignment_records.append(base)

    # Test Set scenario: Train+Val (80%) vs Test (20%) using the final tuned classifier
    for orig_idx in X_trainval.index:
        base = _row_geo(orig_idx)
        base.update({
            "scenario": "Test Set",
            "split": "Training",
            "actual_label": "",
            "predicted_label": "",
            "predicted_probability_high": np.nan,
            "error_type": "",
            "is_misclassified": False,
        })
        fold_assignment_records.append(base)

    test_indices_list = X_test.index.tolist()
    for i, orig_idx in enumerate(test_indices_list):
        base = _row_geo(orig_idx)
        actual = int(y_class_test.iloc[i])
        pred = int(class_test_pred[i])
        prob = float(class_test_proba[i])
        is_correct = (pred == actual)
        if is_correct:
            error = "Correct"
        else:
            error = "False Negative" if actual == 1 else "False Positive"
        base.update({
            "scenario": "Test Set",
            "split": "Test",
            "actual_label": "High" if actual == 1 else "Low/Medium",
            "predicted_label": "High" if pred == 1 else "Low/Medium",
            "predicted_probability_high": round(prob, 4),
            "error_type": error,
            "is_misclassified": not is_correct,
        })
        fold_assignment_records.append(base)

    cv_fold_misclassified = pd.DataFrame(fold_misc_records)
    cv_fold_assignments = pd.DataFrame(fold_assignment_records)

    classifier_model = classifier_pipeline.named_steps["model"]
    classifier_signed = hasattr(classifier_model, "coef_")
    # For tuned or ensemble models, dig into the base estimator for importances
    _feat_model = classifier_model
    if hasattr(_feat_model, "calibrated_classifiers_"):
        _feat_model = _feat_model.calibrated_classifiers_[0].estimator
    if isinstance(_feat_model, VotingClassifier):
        _feat_model = _feat_model.estimators_[1]  # use RF sub-estimator
        classifier_signed = False
    if hasattr(_feat_model, "coef_"):
        classifier_values = _feat_model.coef_[0]
        classifier_signed = True
    elif hasattr(_feat_model, "feature_importances_"):
        classifier_values = _feat_model.feature_importances_
        classifier_signed = False
    else:
        classifier_values = np.zeros(len(MODEL_FEATURES))
        classifier_signed = False

    classifier_importance = pd.DataFrame(
        {
            "feature": [prettify_feature_name(name) for name in MODEL_FEATURES],
            "importance": classifier_values,
        }
    )
    classifier_importance["abs_importance"] = classifier_importance["importance"].abs()
    classifier_importance = classifier_importance.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    regressor_model = regressor_pipeline.named_steps["model"]
    if hasattr(regressor_model, "feature_importances_"):
        regressor_values = regressor_model.feature_importances_
    else:
        regressor_values = np.abs(regressor_model.coef_)
    regressor_importance = pd.DataFrame(
        {
            "feature": [prettify_feature_name(name) for name in MODEL_FEATURES],
            "importance": regressor_values,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    # Use calibrated pipeline for well-calibrated market-strength probabilities
    full_calibrated_clf = CalibratedClassifierCV(best_classifier_model, cv=5, method="isotonic")
    full_calibrated_pipeline = Pipeline([("preprocessor", preprocessor), ("model", full_calibrated_clf)])
    full_calibrated_pipeline.fit(X, y_class)
    market_strength_probability = full_calibrated_pipeline.predict_proba(X)[:, 1]

    classification_test_cases = dataset.loc[
        X_test.index,
        [
            "metro_title",
            "primary_state",
            "region",
            "population",
            "median_income",
            "unemployment_rate",
            "opportunity_score",
        ],
    ].copy()
    classification_test_cases["actual_class"] = y_class_test.astype(int).values
    classification_test_cases["predicted_class"] = class_test_pred.astype(int)
    classification_test_cases["predicted_probability_high"] = class_test_proba
    classification_test_cases["actual_label"] = classification_test_cases["actual_class"].map(
        {0: "Low/Medium", 1: "High"}
    )
    classification_test_cases["predicted_label"] = classification_test_cases["predicted_class"].map(
        {0: "Low/Medium", 1: "High"}
    )
    classification_test_cases["error_type"] = np.where(
        classification_test_cases["actual_class"] == classification_test_cases["predicted_class"],
        "Correct",
        np.where(
            classification_test_cases["actual_class"] == 1,
            "False Negative",
            "False Positive",
        ),
    )

    return {
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "classifier_validation": validation_frame(classifier_validation),
        "regressor_validation": validation_frame(regressor_validation),
        "best_classifier_name": best_classifier_name,
        "best_regressor_name": best_regressor_name,
        "classifier_importance": classifier_importance,
        "classifier_importance_signed": classifier_signed,
        "regressor_importance": regressor_importance,
        "classifier_test_metrics": {
            "accuracy": float(accuracy_score(y_class_test, class_test_pred)),
            "f1": float(f1_score(y_class_test, class_test_pred)),
            "roc_auc": float(roc_auc_score(y_class_test, class_test_proba)),
            "cv_f1_mean": float(class_cv_scores.mean()),
            "cv_f1_std": float(class_cv_scores.std()),
        },
        "market_strength_probability": market_strength_probability,
        "regressor_test_metrics": {
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, reg_test_pred))),
            "mae": float(mean_absolute_error(y_reg_test, reg_test_pred)),
            "r2": float(r2_score(y_reg_test, reg_test_pred)),
            "cv_r2_mean": float(reg_cv_scores.mean()),
            "cv_r2_std": float(reg_cv_scores.std()),
        },
        "classification_report": classification_report(y_class_test, class_test_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_class_test, class_test_pred),
        "classification_test_cases": classification_test_cases,
        "cv_fold_misclassified": cv_fold_misclassified,
        "cv_fold_assignments": cv_fold_assignments,
    }


def save_importance_plot(frame: pd.DataFrame, out_path: Path, title: str, signed: bool) -> None:
    top = frame.head(10).copy()
    if signed:
        top = top.sort_values("importance")
        colors = ["#b22222" if value < 0 else "#2a9d8f" for value in top["importance"]]
    else:
        top = top.sort_values("importance")
        colors = ["#457b9d"] * len(top)

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"], color=colors)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_confusion_matrix_plot(matrix: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Low/Medium", "High"])
    plt.yticks([0, 1], ["Low/Medium", "High"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def normalize_lookup_term(value: str) -> str:
    normalized = str(value).upper().strip()
    normalized = normalized.replace(".", "")
    normalized = normalized.replace(" ST ", " SAINT ")
    normalized = normalized.replace(" ST.", " SAINT")
    normalized = re.sub(r"[^A-Z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def top_factor_labels(factor_scores: Dict[str, float], reverse: bool, limit: int) -> List[str]:
    ordered = sorted(factor_scores.items(), key=lambda item: item[1], reverse=reverse)
    return [label for label, _ in ordered[:limit]]


def join_labels(labels: List[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def classify_entry_strategy(row: pd.Series, thresholds: Dict[str, float]) -> str:
    if (
        row["business_feasibility_score"] >= thresholds["feasibility_high"]
        and row["category_fit_score"] >= thresholds["category_high"]
    ):
        return "Established Leader"
    if (
        row["whitespace_opportunity_score"] >= thresholds["whitespace_high"]
        and row["market_strength_probability"] >= thresholds["market_mid"]
    ):
        return "White-Space Opportunity"
    if (
        row["market_strength_probability"] >= thresholds["market_high"]
        and row["momentum_score"] >= thresholds["momentum_high"]
    ):
        return "Momentum Market"
    if (
        row["category_fit_score"] >= thresholds["category_very_high"]
        and row["market_strength_probability"] < thresholds["market_mid"]
    ):
        return "Niche Cluster Bet"
    return "Selective Expansion Bet"


def build_recommendation_text(row: pd.Series) -> str:
    return (
        f"{row['entry_strategy']}: strongest signals are "
        f"{row['strength_1'].lower()} and {row['strength_2'].lower()}, while the main watchout is "
        f"{row['risk_1'].lower()}."
    )


def create_city_aliases_for_metro(metro_title: str) -> List[str]:
    place_part = metro_title.split(",", 1)[0].strip()
    aliases = {place_part}
    aliases.add(place_part.replace("-", " "))
    aliases.add(place_part.replace(".", ""))
    aliases.add(place_part.replace(".", "").replace("-", " "))

    segments = [
        segment.strip()
        for segment in re.split(r"-|/", place_part)
        if segment.strip()
    ]
    aliases.update(segments)
    aliases.update(segment.replace(".", "").strip() for segment in segments)
    aliases.update(
        segment.replace("Saint", "St").replace(".", "").strip()
        for segment in segments
        if "Saint" in segment
    )
    aliases.update(
        segment.replace("St.", "Saint").replace("St ", "Saint ").strip()
        for segment in segments
        if segment.startswith("St")
    )

    cleaned = []
    for alias in aliases:
        alias = re.sub(r"\s+", " ", alias).strip(" -")
        if alias:
            cleaned.append(alias)
    return sorted(set(cleaned))


def build_city_to_metro_lookup(dataset: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for _, row in dataset.iterrows():
        for alias in create_city_aliases_for_metro(str(row["metro_title"])):
            records.append(
                {
                    "city_alias": alias,
                    "city_alias_norm": normalize_lookup_term(alias),
                    "metro_title": row["metro_title"],
                    "primary_state": row["primary_state"],
                    "region": row["region"],
                    "population": row["population"],
                }
            )

    lookup = pd.DataFrame(records).drop_duplicates(
        subset=["city_alias_norm", "metro_title"]
    )
    lookup = lookup.sort_values(
        ["city_alias_norm", "population", "metro_title"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    lookup["alias_match_rank"] = lookup.groupby("city_alias_norm").cumcount() + 1
    return lookup


def build_business_feasibility_frame(
    dataset: pd.DataFrame,
    market_strength_probability: np.ndarray,
) -> pd.DataFrame:
    market_strength = pd.Series(market_strength_probability, index=dataset.index, dtype=float)
    momentum_score = scale_01(
        dataset["recent_unemployment_change"].max() - dataset["recent_unemployment_change"]
    )
    frames: List[pd.DataFrame] = []

    for business_key, meta in BUSINESS_CATEGORIES.items():
        sector_code = meta["sector_code"]
        emp_share_col = f"emp_share_{sector_code}"
        est_count_col = f"sector_est_{sector_code}"
        est_density_col = f"sector_est_per_10k_{sector_code}"
        sector_share_norm = scale_01(dataset[emp_share_col].fillna(0))
        sector_density_norm = scale_01(dataset[est_density_col].fillna(0))

        category_fit = (
            0.60 * sector_share_norm
            + 0.40 * sector_density_norm
        )
        business_score = 100 * (0.70 * market_strength + 0.30 * category_fit)
        whitespace_base = (
            0.55 * market_strength
            + 0.25 * momentum_score
            + 0.20 * dataset["opportunity_score"]
        )
        whitespace_score = 100 * whitespace_base * (1 - 0.60 * category_fit)

        frame = dataset[
            [
                "metro_title",
                "primary_state",
                "region",
                "population",
                "median_income",
                "unemployment_rate",
                "recent_unemployment_change",
                "opportunity_score",
                "annual_payroll_per_employee",
                "establishments_per_10k",
                "industry_entropy",
                "median_income_norm",
                "inverse_unemployment_norm",
                "annual_payroll_per_employee_norm",
                "establishments_per_10k_norm",
                "industry_entropy_norm",
            ]
        ].copy()
        frame["business_key"] = business_key
        frame["business_type"] = meta["label"]
        frame["sector_code"] = sector_code
        frame["sector_label"] = NAICS_LABELS[sector_code]
        frame["market_strength_probability"] = market_strength
        frame["momentum_score"] = 100 * momentum_score
        frame["sector_employment_share"] = dataset[emp_share_col].fillna(0)
        frame["sector_establishments"] = dataset[est_count_col].fillna(0)
        frame["sector_establishments_per_10k"] = dataset[est_density_col].fillna(0)
        frame["sector_employment_share_norm"] = sector_share_norm
        frame["sector_establishments_per_10k_norm"] = sector_density_norm
        frame["category_fit_score"] = category_fit
        frame["business_feasibility_score"] = business_score
        frame["whitespace_opportunity_score"] = whitespace_score.clip(0, 100)
        frame["business_feasibility_rank"] = (
            frame["business_feasibility_score"].rank(method="min", ascending=False).astype(int)
        )
        frame["business_feasibility_percentile"] = (
            frame["business_feasibility_score"].rank(method="average", pct=True) * 100
        )

        low_cut = float(frame["business_feasibility_score"].quantile(0.40))
        high_cut = float(frame["business_feasibility_score"].quantile(0.70))
        frame["feasibility_band"] = np.where(
            frame["business_feasibility_score"] >= high_cut,
            "High",
            np.where(frame["business_feasibility_score"] >= low_cut, "Medium", "Low"),
        )

        thresholds = {
            "feasibility_high": float(frame["business_feasibility_score"].quantile(0.80)),
            "category_high": float(frame["category_fit_score"].quantile(0.70)),
            "whitespace_high": float(frame["whitespace_opportunity_score"].quantile(0.80)),
            "market_high": float(frame["market_strength_probability"].quantile(0.70)),
            "market_mid": float(frame["market_strength_probability"].quantile(0.50)),
            "momentum_high": float(frame["momentum_score"].quantile(0.70)),
            "category_very_high": float(frame["category_fit_score"].quantile(0.80)),
        }
        frame["entry_strategy"] = frame.apply(classify_entry_strategy, axis=1, thresholds=thresholds)
        frame["strategy_guidance"] = frame["entry_strategy"].map(STRATEGY_GUIDANCE)

        strength_1: List[str] = []
        strength_2: List[str] = []
        strength_3: List[str] = []
        risk_1: List[str] = []
        risk_2: List[str] = []
        recommendation_text: List[str] = []
        factor_summary: List[str] = []

        for _, row in frame.iterrows():
            factor_scores = {
                "Affluent demand": float(row["median_income_norm"]),
                "Low unemployment": float(row["inverse_unemployment_norm"]),
                "Payroll efficiency": float(row["annual_payroll_per_employee_norm"]),
                "Business density": float(row["establishments_per_10k_norm"]),
                "Economic diversity": float(row["industry_entropy_norm"]),
                "Category demand footprint": float(row["sector_employment_share_norm"]),
                "Category location density": float(row["sector_establishments_per_10k_norm"]),
                "Recent labor momentum": float(row["momentum_score"]) / 100.0,
            }
            top_strengths = top_factor_labels(factor_scores, reverse=True, limit=3)
            top_risks = top_factor_labels(factor_scores, reverse=False, limit=2)
            strength_1.append(top_strengths[0])
            strength_2.append(top_strengths[1])
            strength_3.append(top_strengths[2])
            risk_1.append(top_risks[0])
            risk_2.append(top_risks[1])
            factor_summary.append(join_labels(top_strengths))

        frame["strength_1"] = strength_1
        frame["strength_2"] = strength_2
        frame["strength_3"] = strength_3
        frame["risk_1"] = risk_1
        frame["risk_2"] = risk_2
        frame["factor_summary"] = factor_summary
        frame["recommendation_summary"] = frame.apply(build_recommendation_text, axis=1)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def save_region_heatmap(business_frame: pd.DataFrame, out_path: Path) -> None:
    heatmap = business_frame.pivot_table(
        index="region",
        columns="business_type",
        values="business_feasibility_score",
        aggfunc="mean",
    )
    region_order = [region for region in ["Northeast", "West", "Midwest", "South", "Other"] if region in heatmap.index]
    column_order = [meta["label"] for meta in BUSINESS_CATEGORIES.values()]
    heatmap = heatmap.reindex(index=region_order, columns=column_order)

    plt.figure(figsize=(11, 5.5))
    plt.imshow(heatmap.values, cmap="YlGnBu", aspect="auto")
    plt.title("Average Business Feasibility Score by Region and Business Type")
    plt.xticks(range(len(heatmap.columns)), heatmap.columns, rotation=20, ha="right")
    plt.yticks(range(len(heatmap.index)), heatmap.index)
    for row in range(len(heatmap.index)):
        for col in range(len(heatmap.columns)):
            value = heatmap.iloc[row, col]
            if pd.notna(value):
                plt.text(col, row, f"{value:.1f}", ha="center", va="center", color="black")
    plt.colorbar(label="Average Feasibility Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_strategy_mix_plot(business_frame: pd.DataFrame, out_path: Path) -> None:
    strategy_order = [
        "Established Leader",
        "White-Space Opportunity",
        "Momentum Market",
        "Niche Cluster Bet",
        "Selective Expansion Bet",
    ]
    mix = business_frame.pivot_table(
        index="business_type",
        columns="entry_strategy",
        values="metro_title",
        aggfunc="count",
        fill_value=0,
    )
    mix = mix.reindex(columns=[label for label in strategy_order if label in mix.columns], fill_value=0)
    colors = ["#1d3557", "#2a9d8f", "#f4a261", "#6d597a", "#8d99ae"]

    plt.figure(figsize=(11, 6))
    bottom = np.zeros(len(mix))
    for idx, column in enumerate(mix.columns):
        values = mix[column].to_numpy()
        plt.bar(
            mix.index,
            values,
            bottom=bottom,
            label=column,
            color=colors[idx % len(colors)],
        )
        bottom = bottom + values
    plt.title("Phase 3 Entry Strategy Mix by Business Type")
    plt.ylabel("Metro Count")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_whitespace_heatmap(business_frame: pd.DataFrame, out_path: Path) -> None:
    heatmap = business_frame.pivot_table(
        index="region",
        columns="business_type",
        values="whitespace_opportunity_score",
        aggfunc="mean",
    )
    region_order = [region for region in ["Northeast", "West", "Midwest", "South", "Other"] if region in heatmap.index]
    column_order = [meta["label"] for meta in BUSINESS_CATEGORIES.values()]
    heatmap = heatmap.reindex(index=region_order, columns=column_order)

    plt.figure(figsize=(11, 5.5))
    plt.imshow(heatmap.values, cmap="OrRd", aspect="auto")
    plt.title("Average White-Space Opportunity Score by Region and Business Type")
    plt.xticks(range(len(heatmap.columns)), heatmap.columns, rotation=20, ha="right")
    plt.yticks(range(len(heatmap.index)), heatmap.index)
    for row in range(len(heatmap.index)):
        for col in range(len(heatmap.columns)):
            value = heatmap.iloc[row, col]
            if pd.notna(value):
                plt.text(col, row, f"{value:.1f}", ha="center", va="center", color="black")
    plt.colorbar(label="Average White-Space Opportunity Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_business_category_plots(business_frame: pd.DataFrame) -> None:
    save_region_heatmap(
        business_frame,
        OUTPUT_DIR / "business_category_region_heatmap.png",
    )
    save_strategy_mix_plot(
        business_frame,
        OUTPUT_DIR / "entry_strategy_mix_by_business.png",
    )
    save_whitespace_heatmap(
        business_frame,
        OUTPUT_DIR / "whitespace_opportunity_heatmap.png",
    )

    avg_feasibility = (
        business_frame.groupby("business_type", as_index=False)["business_feasibility_score"]
        .mean()
        .sort_values("business_feasibility_score", ascending=False)
    )
    plt.figure(figsize=(10, 5.5))
    plt.bar(
        avg_feasibility["business_type"],
        avg_feasibility["business_feasibility_score"],
        color=["#f4a261", "#2a9d8f", "#1d3557", "#d62828", "#6d597a"],
    )
    plt.title("Average Feasibility Score by Business Type")
    plt.ylabel("Average Feasibility Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_feasibility_by_business_type.png", dpi=200)
    plt.close()

    region_counts = business_frame[["metro_title", "region"]].drop_duplicates()["region"].value_counts()
    metro_count = int(business_frame["metro_title"].nunique())
    plt.figure(figsize=(7, 7))
    plt.pie(
        region_counts.values,
        labels=region_counts.index,
        autopct="%1.0f%%",
        startangle=140,
        colors=["#f4a261", "#457b9d", "#2a9d8f", "#e76f51"],
    )
    plt.title(f"Regional Distribution of the {metro_count}-Metro Sample")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metro_sample_region_pie.png", dpi=200)
    plt.close()

    for business_key, meta in BUSINESS_CATEGORIES.items():
        subset = (
            business_frame[business_frame["business_key"] == business_key]
            .nlargest(15, "business_feasibility_score")
            .sort_values("business_feasibility_score")
        )
        plt.figure(figsize=(11, 7))
        plt.barh(
            subset["metro_title"],
            subset["business_feasibility_score"],
            color=meta["color"],
        )
        plt.title(f"Top 15 Metros for {meta['label']} Feasibility")
        plt.xlabel("Business Feasibility Score")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"top_15_{business_key}_metros.png", dpi=200)
        plt.close()

    restaurant = business_frame[business_frame["business_key"] == "restaurant"].copy()
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        restaurant["sector_establishments_per_10k"],
        restaurant["market_strength_probability"],
        c=restaurant["business_feasibility_score"],
        cmap="YlOrRd",
        s=60,
        alpha=0.85,
    )
    plt.title("Restaurant Feasibility: Category Density vs. Market Strength")
    plt.xlabel("Restaurant-Sector Establishments per 10,000 Residents")
    plt.ylabel("Modeled Market Strength Probability")
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Restaurant Feasibility Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "restaurant_fit_vs_market_strength.png", dpi=200)
    plt.close()


def generate_plots(
    dataset: pd.DataFrame,
    model_outputs: Dict[str, object],
    business_frame: pd.DataFrame,
    cluster_analysis: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    top_metros = dataset.nlargest(15, "opportunity_score").sort_values("opportunity_score")

    plt.figure(figsize=(11, 7))
    plt.barh(top_metros["metro_title"], top_metros["opportunity_score"], color="#2a9d8f")
    plt.title("Top 15 Metros by Opportunity Score")
    plt.xlabel("Opportunity Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_15_metros.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(dataset["opportunity_score"], bins=20, color="#457b9d", edgecolor="white")
    plt.axvline(dataset["opportunity_score"].quantile(TARGET_QUANTILE), color="#b22222", linestyle="--")
    plt.title("Opportunity Score Distribution")
    plt.xlabel("Opportunity Score")
    plt.ylabel("Metro Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "opportunity_score_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        dataset["median_income"],
        dataset["unemployment_rate"],
        c=dataset["opportunity_score"],
        cmap="viridis",
        s=60,
        alpha=0.85,
    )
    plt.title("Median Income vs. Unemployment Rate")
    plt.xlabel("Median Household Income")
    plt.ylabel("Unemployment Rate")
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Opportunity Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "income_vs_unemployment.png", dpi=200)
    plt.close()

    regional_mean = dataset.groupby("region", as_index=False)["opportunity_score"].mean()
    regional_mean = regional_mean.sort_values("opportunity_score", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(regional_mean["region"], regional_mean["opportunity_score"], color="#8d99ae")
    plt.title("Average Opportunity Score by Region")
    plt.ylabel("Opportunity Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regional_opportunity_score.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        dataset["establishments_per_10k"],
        dataset["unemployment_rate"],
        c=np.log10(dataset["population"]),
        cmap="plasma",
        s=60,
        alpha=0.85,
    )
    plt.title("Business Density vs. Unemployment Rate")
    plt.xlabel("Establishments per 10,000 Residents")
    plt.ylabel("Unemployment Rate")
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("log10(Population)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "business_density_vs_unemployment.png", dpi=200)
    plt.close()

    classifier_validation = model_outputs["classifier_validation"].sort_values("validation_f1")
    plt.figure(figsize=(8, 5))
    plt.barh(classifier_validation["model"], classifier_validation["validation_f1"], color="#2a9d8f")
    plt.title("Classifier Validation F1 by Model")
    plt.xlabel("Validation F1")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classifier_model_comparison.png", dpi=200)
    plt.close()

    regressor_validation = model_outputs["regressor_validation"].sort_values("validation_rmse", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.barh(regressor_validation["model"], regressor_validation["validation_rmse"], color="#e76f51")
    plt.title("Regressor Validation RMSE by Model")
    plt.xlabel("Validation RMSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regressor_model_comparison.png", dpi=200)
    plt.close()

    save_importance_plot(
        model_outputs["classifier_importance"],
        OUTPUT_DIR / "classifier_feature_importance.png",
        f"Classifier Importance: {model_outputs['best_classifier_name']}",
        signed=bool(model_outputs["classifier_importance_signed"]),
    )
    save_importance_plot(
        model_outputs["regressor_importance"],
        OUTPUT_DIR / "regressor_feature_importance.png",
        f"Regressor Importance: {model_outputs['best_regressor_name']}",
        signed=False,
    )
    save_confusion_matrix_plot(model_outputs["confusion_matrix"], OUTPUT_DIR / "classifier_confusion_matrix.png")

    region_colors = {
        "Northeast": "#2a9d8f",
        "Midwest": "#f4a261",
        "South": "#e76f51",
        "West": "#457b9d",
        "Other": "#8d99ae",
    }
    plt.figure(figsize=(10, 6))
    for region, group in dataset.groupby("region"):
        plt.scatter(
            group["population"],
            group["opportunity_score"],
            label=region,
            color=region_colors.get(region, "#8d99ae"),
            alpha=0.75,
            s=55,
        )
    plt.xscale("log")
    plt.xlabel("Metro Population (log scale)")
    plt.ylabel("Opportunity Score")
    plt.title("Opportunity Score vs. Metro Population by Region")
    plt.legend(title="Region")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "opportunity_vs_population.png", dpi=200)
    plt.close()

    # PCA + K-Means cluster scatter
    cluster_palette = ["#2a9d8f", "#e76f51", "#457b9d", "#f4a261", "#6d597a", "#8d99ae"]
    unique_clusters = sorted(cluster_analysis["cluster_name"].unique())
    plt.figure(figsize=(11, 7))
    for idx, cname in enumerate(unique_clusters):
        sub = cluster_analysis[cluster_analysis["cluster_name"] == cname]
        plt.scatter(sub["pca_1"], sub["pca_2"],
                    label=cname, color=cluster_palette[idx % len(cluster_palette)],
                    alpha=0.75, s=55)
    plt.title("Metro Clusters in PCA Space (K-Means, k=5)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_pca_scatter.png", dpi=200)
    plt.close()

    # Unemployment trend 2020-2024 for top 10 metros
    trend_cols = ["unemployment_2020", "unemployment_2021", "unemployment_2022",
                  "unemployment_2023", "unemployment_rate"]
    trend_years = [2020, 2021, 2022, 2023, 2024]
    top10_trend = dataset.nlargest(10, "opportunity_score")[["metro_title"] + trend_cols].copy()
    plt.figure(figsize=(12, 6))
    colors_trend = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for idx, (_, row) in enumerate(top10_trend.iterrows()):
        vals = [row[c] for c in trend_cols]
        plt.plot(trend_years, vals, marker="o", linewidth=2,
                 color=colors_trend[idx], label=row["metro_title"][:30])
    plt.title("Unemployment Rate Trend 2020-2024 — Top 10 Opportunity Metros")
    plt.xlabel("Year")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "unemployment_trend_top10.png", dpi=200)
    plt.close()

    # Sensitivity analysis — rank stability of top 20 metros
    top20_sens = sensitivity_df.head(20).copy()
    scenario_cols = [c for c in sensitivity_df.columns
                     if c not in ("metro_title", "rank_mean", "rank_std", "stability")]
    plt.figure(figsize=(12, 8))
    im = plt.imshow(top20_sens[scenario_cols].values, cmap="YlOrRd_r", aspect="auto")
    plt.colorbar(im, label="Rank (lower = better)")
    plt.xticks(range(len(scenario_cols)), scenario_cols, rotation=20, ha="right")
    plt.yticks(range(len(top20_sens)), top20_sens["metro_title"].str[:30], fontsize=8)
    plt.title("Opportunity Rank Sensitivity — Top 20 Metros Across Weight Scenarios")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensitivity_analysis_heatmap.png", dpi=200)
    plt.close()

    # Business category correlation heatmap
    biz_pivot = business_frame.pivot_table(
        index="metro_title", columns="business_type",
        values="business_feasibility_score", aggfunc="first",
    )
    corr_matrix = biz_pivot.corr()
    plt.figure(figsize=(9, 8))
    plt.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    labels = [lab[:22] for lab in corr_matrix.columns]
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    for r in range(len(labels)):
        for c in range(len(labels)):
            plt.text(c, r, f"{corr_matrix.values[r, c]:.2f}",
                     ha="center", va="center", fontsize=7, color="black")
    plt.title("Business Category Score Correlation Across Metros")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "business_category_correlation.png", dpi=200)
    plt.close()

    generate_business_category_plots(business_frame)


def write_reports(
    dataset: pd.DataFrame,
    business_frame: pd.DataFrame,
    city_lookup: pd.DataFrame,
    full_overlap_count: int,
    raw_row_count: int,
    model_outputs: Dict[str, object],
    cluster_analysis: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    sample_scope = metro_sample_scope(len(dataset), full_overlap_count)

    top_metros = dataset.nlargest(10, "opportunity_score")[
        [
            "metro_title",
            "opportunity_score",
            "median_income",
            "unemployment_rate",
            "establishments_per_10k",
            "annual_payroll_per_employee",
        ]
    ].copy()
    top_metros["opportunity_score"] = top_metros["opportunity_score"].round(4)
    top_metros["median_income"] = top_metros["median_income"].round(0)
    top_metros["unemployment_rate"] = top_metros["unemployment_rate"].round(2)
    top_metros["establishments_per_10k"] = top_metros["establishments_per_10k"].round(2)
    top_metros["annual_payroll_per_employee"] = top_metros["annual_payroll_per_employee"].round(2)

    dataset.to_csv(OUTPUT_DIR / "model_dataset_selected_metros.csv", index=False)
    top_metros.to_csv(OUTPUT_DIR / "top_10_metros.csv", index=False)
    business_frame.to_csv(OUTPUT_DIR / "business_feasibility_scores.csv", index=False)
    city_lookup.to_csv(OUTPUT_DIR / "city_to_metro_lookup.csv", index=False)

    recommendation_engine = business_frame[
        [
            "metro_title",
            "primary_state",
            "region",
            "business_key",
            "business_type",
            "business_feasibility_score",
            "business_feasibility_rank",
            "feasibility_band",
            "entry_strategy",
            "strategy_guidance",
            "market_strength_probability",
            "category_fit_score",
            "whitespace_opportunity_score",
            "momentum_score",
            "recommendation_summary",
            "strength_1",
            "strength_2",
            "strength_3",
            "risk_1",
            "risk_2",
            "recent_unemployment_change",
            "population",
            "median_income",
            "unemployment_rate",
        ]
    ].copy()
    recommendation_engine.to_csv(OUTPUT_DIR / "business_recommendation_engine.csv", index=False)
    classification_test_cases = model_outputs["classification_test_cases"].copy()
    classification_test_cases.to_csv(OUTPUT_DIR / "classification_test_cases.csv", index=False)
    misclassified_cases = classification_test_cases[
        classification_test_cases["error_type"] != "Correct"
    ].copy()
    misclassified_cases.to_csv(OUTPUT_DIR / "misclassified_cases.csv", index=False)
    model_outputs["cv_fold_misclassified"].to_csv(OUTPUT_DIR / "cv_fold_misclassified.csv", index=False)
    model_outputs["cv_fold_assignments"].to_csv(OUTPUT_DIR / "cv_fold_assignments.csv", index=False)

    top_by_business = (
        business_frame.sort_values("business_feasibility_score", ascending=False)
        .groupby("business_type", group_keys=False)
        .head(10)
        .copy()
    )
    top_by_business.to_csv(OUTPUT_DIR / "top_10_metros_by_business.csv", index=False)
    established_markets = (
        business_frame[business_frame["entry_strategy"] == "Established Leader"]
        .sort_values("business_feasibility_score", ascending=False)
        .groupby("business_type", group_keys=False)
        .head(10)
        .copy()
    )
    established_markets.to_csv(OUTPUT_DIR / "top_10_established_markets_by_business.csv", index=False)
    whitespace_markets = (
        business_frame[business_frame["entry_strategy"] == "White-Space Opportunity"]
        .sort_values("whitespace_opportunity_score", ascending=False)
        .groupby("business_type", group_keys=False)
        .head(10)
        .copy()
    )
    whitespace_markets.to_csv(OUTPUT_DIR / "top_10_whitespace_markets_by_business.csv", index=False)

    cluster_enriched = dataset[["metro_title"]].merge(
        cluster_analysis[["metro_title", "cluster", "cluster_name"]], on="metro_title", how="left"
    )
    full_metro_ranking = dataset.sort_values("opportunity_score", ascending=False).reset_index(drop=True).copy()
    full_metro_ranking.insert(0, "rank", range(1, len(full_metro_ranking) + 1))
    full_metro_ranking = full_metro_ranking.merge(
        cluster_enriched, on="metro_title", how="left"
    )
    full_ranking_cols = [
        "rank", "metro_title", "primary_state", "region",
        "opportunity_score", "median_income", "real_median_income",
        "cost_of_living_index", "housing_cost_index",
        "unemployment_rate", "population", "establishments_per_10k",
        "annual_payroll_per_employee", "industry_entropy",
        "unemployment_trend_slope", "state_business_climate_norm",
        "lat", "lon",
        "high_opportunity", "cluster_name",
    ]
    # Drop lat/lon from the export if the gazetteer was missing
    full_ranking_cols = [c for c in full_ranking_cols if c in full_metro_ranking.columns]
    full_metro_ranking[full_ranking_cols].to_csv(OUTPUT_DIR / "full_metro_ranking.csv", index=False)

    cluster_analysis.to_csv(OUTPUT_DIR / "cluster_analysis.csv", index=False)
    sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_analysis.csv", index=False)

    metro_business_pivot = business_frame.pivot_table(
        index=["metro_title", "primary_state", "region", "population", "opportunity_score"],
        columns="business_key",
        values="business_feasibility_score",
        aggfunc="first",
    ).reset_index()
    metro_business_pivot.columns.name = None
    metro_business_pivot.to_csv(OUTPUT_DIR / "metro_business_pivot.csv", index=False)

    model_outputs["classifier_validation"].to_csv(
        OUTPUT_DIR / "classifier_validation_results.csv", index=False
    )
    model_outputs["regressor_validation"].to_csv(
        OUTPUT_DIR / "regressor_validation_results.csv", index=False
    )
    model_outputs["classifier_importance"].to_csv(
        OUTPUT_DIR / "classifier_feature_importance.csv", index=False
    )
    model_outputs["regressor_importance"].to_csv(
        OUTPUT_DIR / "regressor_feature_importance.csv", index=False
    )

    classifier_metrics = model_outputs["classifier_test_metrics"]
    regressor_metrics = model_outputs["regressor_test_metrics"]
    split_sizes = model_outputs["split_sizes"]

    with (OUTPUT_DIR / "classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Best classifier: {model_outputs['best_classifier_name']}\n")
        handle.write(
            f"Split: {split_sizes['train']}/{split_sizes['validation']}/{split_sizes['test']} "
            "(train/validation/test)\n"
        )
        handle.write(
            f"Test Accuracy: {classifier_metrics['accuracy']:.4f}\n"
            f"Test F1: {classifier_metrics['f1']:.4f}\n"
            f"Test ROC-AUC: {classifier_metrics['roc_auc']:.4f}\n"
            f"5-fold CV F1: {classifier_metrics['cv_f1_mean']:.4f} +/- {classifier_metrics['cv_f1_std']:.4f}\n\n"
        )
        handle.write(model_outputs["classification_report"])

    with (OUTPUT_DIR / "regression_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Best regressor: {model_outputs['best_regressor_name']}\n")
        handle.write(
            f"Target: 2024 metropolitan unemployment rate\n"
            f"Test RMSE: {regressor_metrics['rmse']:.4f}\n"
            f"Test MAE: {regressor_metrics['mae']:.4f}\n"
            f"Test R2: {regressor_metrics['r2']:.4f}\n"
            f"5-fold CV R2: {regressor_metrics['cv_r2_mean']:.4f} +/- {regressor_metrics['cv_r2_std']:.4f}\n"
        )

    business_lines = [
        "# Business-Specific Feasibility Summary",
        "",
        "Broad business categories are scored as:",
        "- 70% modeled metro market strength",
        "- 30% category fit from CBP sector presence",
        "",
    ]

    for business_key, meta in BUSINESS_CATEGORIES.items():
        subset = business_frame[business_frame["business_key"] == business_key].nlargest(
            5, "business_feasibility_score"
        )
        business_lines.append(f"## {meta['label']}")
        for _, row in subset.iterrows():
            business_lines.append(
                f"- {row['metro_title']}: score={row['business_feasibility_score']:.1f}, "
                f"market_strength={row['market_strength_probability']:.3f}, "
                f"sector_share={row['sector_employment_share']:.3f}, "
                f"sector_est_per_10k={row['sector_establishments_per_10k']:.2f}"
            )
        business_lines.append("")

    with (OUTPUT_DIR / "business_feasibility_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(business_lines))

    metadata = {
        "selected_metros": int(len(dataset)),
        "overlap_metros": int(full_overlap_count),
        "raw_cbp_rows_used": int(raw_row_count),
        "sample_scope": sample_scope,
        "primary_state_labels": int(dataset["primary_state"].nunique()),
        "business_types_supported": int(business_frame["business_type"].nunique()),
    }
    with (OUTPUT_DIR / "project_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    phase3_lines = [
        "# Phase 3 Recommendation Layer Summary",
        "",
        "## What was added",
        "- Entry strategy labeling for each metro-business combination",
        "- White-space opportunity scoring to surface strong markets with expansion room",
        "- Momentum scoring based on recent unemployment change",
        "- Auto-generated strengths, risks, and recommendation text for each recommendation row",
        "",
        "## Strategy definitions",
        "- Established Leader: high feasibility with high existing category fit",
        "- White-Space Opportunity: strong metro fundamentals with room for category expansion",
        "- Momentum Market: good fundamentals plus stronger recent labor-market movement",
        "- Niche Cluster Bet: strong category presence but thinner overall metro support",
        "- Selective Expansion Bet: usable market, but not a top-priority entry pick",
        "",
        "## Strategy mix by business type",
    ]

    strategy_mix = (
        business_frame.groupby(["business_type", "entry_strategy"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for business_type, row in strategy_mix.iterrows():
        strategy_bits = ", ".join(f"{label}={int(value)}" for label, value in row.items() if value > 0)
        phase3_lines.append(f"- {business_type}: {strategy_bits}")

    phase3_lines.extend(
        [
            "",
            "## White-space leaders by business type",
        ]
    )
    for business_key, meta in BUSINESS_CATEGORIES.items():
        subset = business_frame[business_frame["business_key"] == business_key].nlargest(
            3, "whitespace_opportunity_score"
        )
        winners = ", ".join(
            f"{row['metro_title']} ({row['whitespace_opportunity_score']:.1f})"
            for _, row in subset.iterrows()
        )
        phase3_lines.append(f"- {meta['label']}: {winners}")

    phase3_lines.extend(
        [
            "",
            "## How to use it",
            "- Use Established Leader when the goal is safer, proven entry.",
            "- Use White-Space Opportunity when the goal is to find stronger markets with less category saturation.",
            "- Use Momentum Market when recent labor-market improvement matters.",
            "- Use the scenario lab in the app for transparent what-if adjustments.",
        ]
    )

    with (OUTPUT_DIR / "phase3_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(phase3_lines))

    summary_lines = [
        f"# {PROJECT_TITLE}",
        "",
        f"## {PROJECT_SUBTITLE}",
        "",
        "## Executive Summary",
        f"- Official CBP raw rows available: 577,424 across 925 metro/micro areas",
        f"- Official ACS metro areas available: 393 metro areas",
        f"- CBP + ACS + BLS overlap: {full_overlap_count} metropolitan areas",
        f"- Final analysis sample: {sample_scope}",
        f"- Raw CBP rows used before aggregation: {raw_row_count}",
        f"- Positive class definition: top {int((1 - TARGET_QUANTILE) * 100)}% by opportunity score",
        "",
        "## Opportunity Score",
        "- 25% median household income",
        "- 25% inverse unemployment rate",
        "- 20% annual payroll per employee",
        "- 15% establishments per 10,000 residents",
        "- 15% industry entropy (business diversity)",
        "",
        "## Modeling Setup",
        "- Unit of analysis: metropolitan area",
        "- Classification target: high-opportunity metro",
        "- Regression target: 2024 unemployment rate",
        "- Features: CBP structure metrics and 2-digit industry employment shares",
        "- Split: 60/20/20 train/validation/test",
        "",
        "## Business-Specific Feasibility Layer",
        "- Broad business types supported: Restaurant / Food Service, Retail Store, Health Clinic / Care Service, Professional Services Firm, Salon / Repair / Personal Services",
        "- Business feasibility score = 70% modeled metro market strength + 30% category fit from CBP sector presence",
        "- Category fit is based on sector employment share and sector establishments per 10,000 residents",
        "- Additional weekly update: recent unemployment change (2024 - 2023) added to the dataset for metro lookup and future model experiments",
        f"- City-to-metro lookup aliases generated: {int(city_lookup['city_alias_norm'].nunique())}",
        "",
        "## Best Model Results",
        f"- Best classifier: {model_outputs['best_classifier_name']}",
        f"- Test Accuracy: {classifier_metrics['accuracy']:.4f}",
        f"- Test F1: {classifier_metrics['f1']:.4f}",
        f"- Test ROC-AUC: {classifier_metrics['roc_auc']:.4f}",
        f"- 5-fold CV F1: {classifier_metrics['cv_f1_mean']:.4f} +/- {classifier_metrics['cv_f1_std']:.4f}",
        f"- Best regressor: {model_outputs['best_regressor_name']}",
        f"- Test RMSE: {regressor_metrics['rmse']:.4f}",
        f"- Test MAE: {regressor_metrics['mae']:.4f}",
        f"- Test R2: {regressor_metrics['r2']:.4f}",
        f"- 5-fold CV R2: {regressor_metrics['cv_r2_mean']:.4f} +/- {regressor_metrics['cv_r2_std']:.4f}",
        "",
        "## Top Business-Type Examples",
    ]

    for business_key, meta in BUSINESS_CATEGORIES.items():
        top_row = business_frame[business_frame["business_key"] == business_key].nlargest(
            1, "business_feasibility_score"
        ).iloc[0]
        summary_lines.append(
            f"- {meta['label']}: {top_row['metro_title']} with score={top_row['business_feasibility_score']:.1f}"
        )

    summary_lines.extend(
        [
            "",
        "## Top 10 Metros",
        ]
    )

    for _, row in top_metros.iterrows():
        summary_lines.append(
            f"- {row['metro_title']}: score={row['opportunity_score']:.4f}, "
            f"income={int(row['median_income'])}, unemployment={row['unemployment_rate']:.2f}, "
            f"est/10k={row['establishments_per_10k']:.2f}"
        )

    summary_lines.extend(
        [
        "",
        "## Limitations",
        "- CBP is 2022 while ACS and BLS are 2024, so the study has a small time mismatch.",
        "- This is a metro-level market study, not a business-level causal model.",
        "- The opportunity index is a decision-support ranking tool, not a single ground-truth outcome.",
        "- Phase 3 recommendation strategies are rule-based extensions on top of the Phase 2 model outputs.",
    ]
    )

    with (OUTPUT_DIR / "phase2_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))

    n_features = len(MODEL_FEATURES)
    n_categories = int(business_frame['business_type'].nunique())
    final_report_lines = [
        f"# {PROJECT_TITLE}",
        "## Final Submission Report",
        "",
        "---",
        "",
        "## Project Status",
        "- **Status:** complete and submission-ready",
        f"- **Final metro sample:** {sample_scope}",
        f"- **Raw CBP rows used:** {raw_row_count:,}",
        f"- **Business categories supported:** {n_categories}",
        f"- **Total model features:** {n_features}",
        "",
        "---",
        "",
        "## Data Stack (4 official sources)",
        "",
        "| Dataset | Year | Provider |",
        "|---------|------|----------|",
        f"| County Business Patterns (CBP) | **{CBP_YEAR}** | U.S. Census Bureau |",
        "| American Community Survey (ACS) | 2024 | U.S. Census Bureau |",
        "| BLS Metro Unemployment | 2020-2024 | Bureau of Labor Statistics |",
        "| **BEA Regional Price Parities** | **2024** | Bureau of Economic Analysis |",
        "",
        "---",
        "",
        "## Modeling Summary",
        "",
        "### Classification (high-opportunity metros)",
        f"- **Best classifier:** {model_outputs['best_classifier_name']}",
        f"- **Test Accuracy:** {classifier_metrics['accuracy']:.4f}",
        f"- **Test F1:** {classifier_metrics['f1']:.4f}",
        f"- **Test ROC-AUC:** {classifier_metrics['roc_auc']:.4f}",
        "",
        "### Regression (unemployment rate)",
        f"- **Best regressor:** {model_outputs['best_regressor_name']}",
        f"- **Test RMSE:** {regressor_metrics['rmse']:.4f}",
        f"- **Test MAE:** {regressor_metrics['mae']:.4f}",
        f"- **Test R2:** {regressor_metrics['r2']:.4f}",
        "",
        "---",
        "",
        "## Opportunity Score (Phase 5 formula)",
        "",
        "```",
        "0.25 * real_median_income_norm     (BEA-RPP-adjusted)",
        "0.20 * inverse_unemployment_norm",
        "0.15 * annual_payroll_per_employee_norm",
        "0.15 * establishments_per_10k_norm",
        "0.15 * industry_entropy_norm",
        "0.10 * affordability_norm           (cost-of-living penalty)",
        "```",
        "",
        "Cost-of-living adjustment shifted top rankings: SF #2->#4, Boston top-5->#13,",
        "Midland TX -> #2, Sioux Falls and Bismarck entered top 6.",
        "",
        "---",
        "",
        "## Phase Delivery Summary",
        "- **Phase 1**: 32-feature pipeline, payroll/labour ratios, 6 categories",
        "- **Phase 2**: 4-source merge, opportunity score, classifier+regressor",
        "- **Phase 3**: entry-strategy labels, white-space scoring, city aliases, what-if lab",
        "- **Phase 4**: GridSearchCV tuning, VotingClassifier, calibrated probs, K-Means",
        "  clustering, PCA, sensitivity analysis, 9 categories, 8-tab BI dashboard",
        "- **Phase 5**: BEA RPP cost-of-living + housing, CBP 2022->2023 refresh, lat/lon",
        "  bubble map, executive PowerPoint generator, dashboard cost-of-living surfacing",
        "",
        "---",
        "",
        "## Key Files",
        "- `README.md` - project overview and usage",
        "- `CHANGELOG.md` - full history across 5 phases",
        "- `capstone_pipeline.py` - core pipeline (~2,200 lines)",
        "- `app.py` - Streamlit BI dashboard, 8 interactive tabs",
        "- `exec_report.py` - executive PowerPoint and PDF generator",
        "- `main.py` - entry point",
        "- `outputs/` - 50+ generated CSVs, PNGs, and reports",
        "- `outputs/executive_summary.pptx` - submission-ready slide deck",
        "- `outputs/executive_summary.pdf` - one-page PDF summary",
        "",
        "---",
        "",
        "## Limitations",
        f"- CBP is {CBP_YEAR}; ACS, BLS, and BEA RPP are 2024 (intentional given staggered release)",
        "- Metro-level feasibility, not a business-level causal model",
        "- Opportunity score is decision-support; sensitivity analysis exposes weight choices",
        "",
        "---",
        "",
        "## How to Reproduce",
        "```bash",
        "pip install -r requirements.txt",
        "python3 main.py",
        "streamlit run app.py",
        "```",
    ]

    with (BASE_DIR / "FINAL_REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(final_report_lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading official CBP {CBP_YEAR} data ({CBP_ZIP_PATH.name})...")
    cbp = load_cbp_raw(CBP_ZIP_PATH)

    print("Loading official ACS metro data...")
    acs = load_acs_metro(ACS_JSON_PATH)

    print("Loading official BLS metro unemployment data...")
    bls = load_bls_metro(BLS_XLSX_PATH)

    print("Loading BEA Regional Price Parities (cost-of-living + housing)...")
    bea_rpp = load_bea_rpp(BEA_RPP_PATH, year=2024)
    if bea_rpp.empty:
        print("  · BEA RPP file not found — using neutral defaults (100).")
    else:
        print(f"  · BEA RPP loaded for {len(bea_rpp):,} MSAs.")

    print("Loading Census CBSA gazetteer (lat/lon centroids)...")
    centroids = load_metro_centroids(GAZETTEER_PATH)
    if centroids.empty:
        print("  · Gazetteer not found — bubble map will be skipped.")
    else:
        print(f"  · Centroids loaded for {len(centroids):,} CBSAs.")

    selection_label = (
        "all overlapping metros"
        if METRO_SAMPLE_LIMIT is None
        else f"top {METRO_SAMPLE_LIMIT} metros from the three-source overlap"
    )
    print(f"Selecting {selection_label}...")
    selected_metros, overlap_count = build_selected_metros(cbp, acs, bls)

    print("Engineering CBP structural features...")
    cbp_features, raw_row_count = engineer_cbp_features(cbp, selected_metros["cbsa"])

    print("Building metro-level modeling dataset...")
    dataset = build_model_dataset(selected_metros, cbp_features, bea_rpp=bea_rpp, centroids=centroids)

    print("Training and evaluating models...")
    model_outputs = evaluate_models(dataset)

    print("Building business-specific feasibility dataset...")
    business_frame = build_business_feasibility_frame(
        dataset,
        model_outputs["market_strength_probability"],
    )

    print("Building city-to-metro lookup aliases...")
    city_lookup = build_city_to_metro_lookup(dataset)

    print("Clustering metros (K-Means, k=5)...")
    cluster_analysis, cluster_profiles, pca_variance = cluster_metros(dataset)

    print("Computing opportunity-score sensitivity analysis...")
    sensitivity_df = compute_sensitivity(dataset)

    print("Generating figures and reports...")
    generate_plots(dataset, model_outputs, business_frame, cluster_analysis, sensitivity_df)
    write_reports(
        dataset,
        business_frame,
        city_lookup,
        overlap_count,
        raw_row_count,
        model_outputs,
        cluster_analysis,
        sensitivity_df,
    )

    # ── Executive deliverables (PPTX + PDF) ──────────────────────────────────
    print("Building executive PowerPoint and PDF (via exec_report)...")
    try:
        from exec_report import build_executive_pptx, build_executive_pdf
        pptx_path = build_executive_pptx()
        pdf_path  = build_executive_pdf()
        print(f"  · {pptx_path.relative_to(BASE_DIR)}")
        print(f"  · {pdf_path.relative_to(BASE_DIR)}")
    except ImportError as e:
        print(f"  · skipped (python-pptx not installed): {e}")
    except Exception as e:
        print(f"  · failed: {e}")

    print("\nMetro capstone pipeline completed.")
    print(f"Selected metros: {len(dataset)}")
    print(f"Business categories: {len(BUSINESS_CATEGORIES)}")
    print(f"Raw CBP rows used: {raw_row_count}")
    print(f"Best classifier: {model_outputs['best_classifier_name']}")
    print(f"Best regressor: {model_outputs['best_regressor_name']}")
    print(f"Cluster PCA variance explained: {pca_variance[0]:.1%} / {pca_variance[1]:.1%}")


if __name__ == "__main__":
    main()
