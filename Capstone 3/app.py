"""
Metropolitan Business Feasibility Prediction System
====================================================
Interactive BI Dashboard (Plotly + Streamlit)

Six dashboards:
  1. Executive Overview   — KPIs, choropleth, top performers
  2. Market Explorer      — drill into a single metro
  3. Business Opportunity — per-business analytics + map
  4. What-If Scenario Lab — live scenario simulation
  5. Compare Metros       — side-by-side multi-metro comparison
  6. Model Insights       — feature importance, validation, errors

Global sidebar filters cross-filter aggregate views (PowerBI-style).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NAICS_LABELS = {
    "11": "Agriculture", "21": "Mining and Oil/Gas", "22": "Utilities",
    "23": "Construction", "31": "Manufacturing", "42": "Wholesale Trade",
    "44": "Retail Trade", "48": "Transportation and Warehousing",
    "51": "Information", "52": "Finance and Insurance", "53": "Real Estate",
    "54": "Professional Services", "55": "Management of Companies",
    "56": "Administrative Support", "61": "Educational Services",
    "62": "Health Care", "71": "Arts and Recreation",
    "72": "Accommodation and Food", "81": "Other Services",
    "99": "Unclassified",
}

# ── Modern Indigo palette ────────────────────────────────────────────────────
# Clean, professional, BI-grade. Inspired by Linear / Stripe / modern dashboards.
# Constants reused throughout the chart code — change here, propagates everywhere.

# Slate / surface scale
SAND_50   = "#f9fafb"   # page background
SAND_100  = "#f3f4f6"   # secondary surface
SAND_200  = "#e5e7eb"   # borders
SAND_300  = "#cbd5e1"   # muted accents
SAND_400  = "#94a3b8"   # disabled / placeholder
SAND_500  = "#64748b"   # muted text
SAND_600  = "#475569"   # body secondary
SAND_700  = "#334155"   # body
SAND_800  = "#0f172a"   # primary text / heading

# Indigo (primary brand)
TERRA_400 = "#818cf8"   # indigo-400 — light highlight
TERRA_500 = "#4f46e5"   # indigo-600 — PRIMARY accent
TERRA_600 = "#4338ca"   # indigo-700 — hover / strong
TERRA_700 = "#3730a3"   # indigo-800 — deep

# Emerald (success / positive)
SAGE_400  = "#34d399"   # emerald-400
SAGE_500  = "#10b981"   # emerald-500 — SUCCESS
SAGE_600  = "#059669"   # emerald-600 — deep success

# Aliases for legibility
INK_900   = "#0f172a"   # primary text
INK_700   = "#334155"   # body text
INK_500   = "#64748b"   # muted text

# Region palette — distinct, harmonious, BI-friendly
REGION_PALETTE = {
    "Northeast": "#4f46e5",   # indigo
    "Midwest":   "#f59e0b",   # amber
    "South":     "#ef4444",   # red — sun-baked South
    "West":      "#10b981",   # emerald
    "Other":     "#64748b",   # slate
}

# Strategy palette — clear semantic mapping
STRATEGY_PALETTE = {
    "Established Leader":      "#1e1b4b",   # indigo-950 — deep, established
    "White-Space Opportunity": "#10b981",   # emerald — fresh ground
    "Momentum Market":         "#f59e0b",   # amber — heat
    "Niche Cluster Bet":       "#06b6d4",   # cyan — focused
    "Selective Expansion Bet": "#64748b",   # slate — measured
}

# Plotly sequential scale — light → indigo → midnight
DESERT_SCALE = [
    [0.00, "#f1f5f9"],   # slate-100
    [0.20, "#c7d2fe"],   # indigo-200
    [0.40, "#818cf8"],   # indigo-400
    [0.60, "#6366f1"],   # indigo-500
    [0.80, "#4338ca"],   # indigo-700
    [1.00, "#1e1b4b"],   # indigo-950
]

# Plotly diverging — red (low) → amber (mid) → emerald (high)
DESERT_DIVERGING = [
    [0.00, "#ef4444"],   # red-500
    [0.50, "#f59e0b"],   # amber-500
    [1.00, "#10b981"],   # emerald-500
]

# Reversed diverging — emerald (low) → amber → red (high; e.g. RMSE/error)
DESERT_DIVERGING_R = [
    [0.00, "#10b981"],
    [0.50, "#f59e0b"],
    [1.00, "#ef4444"],
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def extract_metric(report_text: str, label: str) -> str:
    m = re.search(rf"{re.escape(label)}:\s*(.+)", report_text)
    return m.group(1).strip() if m else "n/a"


@st.cache_data(show_spinner=False)
def load_outputs() -> dict:
    dataset = pd.read_csv(OUTPUT_DIR / "model_dataset_selected_metros.csv")
    dataset["high_opportunity_metro"] = dataset["high_opportunity"].astype(int)
    dataset["opportunity_rank"] = (
        dataset["opportunity_score"].rank(method="min", ascending=False).astype(int)
    )

    metadata = json.loads((OUTPUT_DIR / "project_metadata.json").read_text("utf-8"))
    business_scores = pd.read_csv(OUTPUT_DIR / "business_feasibility_scores.csv")
    recommendation_engine = pd.read_csv(OUTPUT_DIR / "business_recommendation_engine.csv")
    full_metro_ranking = pd.read_csv(OUTPUT_DIR / "full_metro_ranking.csv")
    metro_business_pivot = pd.read_csv(OUTPUT_DIR / "metro_business_pivot.csv")
    city_lookup = pd.read_csv(OUTPUT_DIR / "city_to_metro_lookup.csv")

    classifier_validation = pd.read_csv(OUTPUT_DIR / "classifier_validation_results.csv")
    regressor_validation = pd.read_csv(OUTPUT_DIR / "regressor_validation_results.csv")
    classifier_importance = pd.read_csv(OUTPUT_DIR / "classifier_feature_importance.csv")
    regressor_importance = pd.read_csv(OUTPUT_DIR / "regressor_feature_importance.csv")
    misclassified_cases = pd.read_csv(OUTPUT_DIR / "misclassified_cases.csv")
    classification_test_cases = pd.read_csv(OUTPUT_DIR / "classification_test_cases.csv")
    _cv_fold_path = OUTPUT_DIR / "cv_fold_misclassified.csv"
    cv_fold_misclassified = pd.read_csv(_cv_fold_path) if _cv_fold_path.exists() else pd.DataFrame()
    _cv_assign_path = OUTPUT_DIR / "cv_fold_assignments.csv"
    cv_fold_assignments = pd.read_csv(_cv_assign_path) if _cv_assign_path.exists() else pd.DataFrame()
    cluster_analysis = pd.read_csv(OUTPUT_DIR / "cluster_analysis.csv")
    sensitivity_df = pd.read_csv(OUTPUT_DIR / "sensitivity_analysis.csv")

    return {
        "dataset": dataset,
        "metadata": metadata,
        "business_scores": business_scores,
        "recommendation_engine": recommendation_engine,
        "full_metro_ranking": full_metro_ranking,
        "metro_business_pivot": metro_business_pivot,
        "city_lookup": city_lookup,
        "classifier_validation": classifier_validation,
        "regressor_validation": regressor_validation,
        "classifier_importance": classifier_importance,
        "regressor_importance": regressor_importance,
        "misclassified_cases": misclassified_cases,
        "classification_test_cases": classification_test_cases,
        "cv_fold_misclassified": cv_fold_misclassified,
        "cv_fold_assignments": cv_fold_assignments,
        "cluster_analysis": cluster_analysis,
        "sensitivity_df": sensitivity_df,
        "classification_report": read_text(OUTPUT_DIR / "classification_report.txt"),
        "regression_report": read_text(OUTPUT_DIR / "regression_report.txt"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def kpi_card(label: str, value: str, delta: str | None = None):
    if delta is not None:
        st.metric(label, value, delta)
    else:
        st.metric(label, value)


def scale_value(value: float, series: pd.Series) -> float:
    lo, hi = float(series.min()), float(series.max())
    if hi == lo:
        return 0.0
    return (max(lo, min(hi, float(value))) - lo) / (hi - lo)


def clean_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
        xaxis=dict(gridcolor="#eef2f7", linecolor="#e2e8f0",
                   tickfont=dict(size=11, color="#64748b"), zeroline=False),
        yaxis=dict(gridcolor="#eef2f7", linecolor="#e2e8f0",
                   tickfont=dict(size=11, color="#64748b"), zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(size=11, color="#334155")),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                        font=dict(family="Inter, sans-serif", size=12, color="#0f172a")),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Design system — injected once after st.set_page_config
# ─────────────────────────────────────────────────────────────────────────────

APP_CSS = """
<style>
/* ── Google Font ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Force light desert theme regardless of OS preference ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.stApp, .main {
    background-color: #f9fafb !important;
    color: #0f172a !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 0% 0%, rgba(79,70,229,0.10), transparent 45%),
        radial-gradient(circle at 100% 0%, rgba(99,102,241,0.12), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(16,185,129,0.06), transparent 40%),
        #f9fafb !important;
}

/* ── Base text ───────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.main p, .main span, .main label, .main li, .main div {
    color: #0f172a;
}

/* ── Hide Streamlit chrome ───────────────────── */
#MainMenu, footer, header { visibility: hidden; height: 0 !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Main container ──────────────────────────── */
.main .block-container {
    padding: 0.5rem 2rem 2rem !important;
    max-width: 1600px !important;
}

/* ── Page fade-in ────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
section.main > div { animation: fadeIn 0.45s ease-out; }

/* ── Metric cards ────────────────────────────── */
[data-testid="metric-container"], div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.3rem !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.05) !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="metric-container"]::before, div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4f46e5, #818cf8, #6366f1);
    border-radius: 14px 14px 0 0;
}
[data-testid="metric-container"]:hover, div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 26px rgba(15,23,42,0.12) !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    color: #334155 !important;
}

/* ── Tabs ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px !important;
    background: #f3f4f6 !important;
    padding: 5px !important;
    border-radius: 12px !important;
    border-bottom: none !important;
    flex-wrap: wrap !important;
    border: 1px solid #e5e7eb !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 7px 14px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    color: #334155 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.18s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #e5e7eb !important;
    color: #0f172a !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #4f46e5 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.08) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── Section headings ────────────────────────── */
h1, h2, h3, h4, h5, h6,
.main h1, .main h2, .main h3, .main h4 {
    color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
}
.main h2 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    margin-top: 1.5rem !important;
    padding-bottom: 0.55rem !important;
    border-bottom: 2px solid #e5e7eb !important;
}
.main h3 {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin-top: 1.3rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid #e5e7eb !important;
}

/* ── Primary buttons ─────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.38) !important;
    filter: brightness(1.04);
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ── Download buttons ────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.38) !important;
    filter: brightness(1.04);
}

/* ── Form inputs ─────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 10px !important;
    border-color: #e5e7eb !important;
    background-color: #fff !important;
    font-size: 0.88rem !important;
    color: #0f172a !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
}

/* ── Expanders ───────────────────────────────── */
.streamlit-expanderHeader {
    background: #f8fafc !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1px solid #e5e7eb !important;
    font-size: 0.88rem !important;
    color: #0f172a !important;
}

/* ── Dataframe ───────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.07) !important;
    border: 1px solid #e5e7eb !important;
}

/* ── Alerts ──────────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    background-color: #f8fafc !important;
    border: 1px solid #e5e7eb !important;
    color: #0f172a !important;
}

/* ── Dividers ────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #e5e7eb !important;
    margin: 1.5rem 0 !important;
}

/* ── Plotly chart containers ─────────────────── */
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.05) !important;
    transition: box-shadow 0.22s ease, transform 0.22s ease !important;
    padding: 0.5rem !important;
}
.stPlotlyChart:hover, [data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 6px 22px rgba(15,23,42,0.10) !important;
    transform: translateY(-1px) !important;
}
/* Hide Plotly modebar (camera/zoom icons) by default — show on hover */
.modebar { opacity: 0 !important; transition: opacity 0.2s !important; }
.stPlotlyChart:hover .modebar { opacity: 0.55 !important; }

/* ── Sidebar — clean light surface with indigo accents ─────── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
    background:
        linear-gradient(180deg, #fdfcff 0%, #f5f3ff 100%) !important;
    border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li {
    color: #334155 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
}
/* Widget labels — strong indigo for high contrast on light bg */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] div {
    color: #4338ca !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}
/* Reset filter button — outlined indigo */
section[data-testid="stSidebar"] .stButton > button {
    background: #ffffff !important;
    border: 1px solid #c7d2fe !important;
    color: #4338ca !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04) !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #eef2ff !important;
    border-color: #818cf8 !important;
    color: #312e81 !important;
    transform: none !important;
    box-shadow: 0 2px 6px rgba(79,70,229,0.15) !important;
    filter: none !important;
}
section[data-testid="stSidebar"] hr,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
    border-color: #e5e7eb !important;
}
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stCheckbox label p {
    color: #334155 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaption"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #64748b !important;
}
/* MultiSelect tags (Midwest x, Northeast x, ...) */
section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #eef2ff !important;
    color: #4338ca !important;
    border: 1px solid #c7d2fe !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] [role="button"] {
    color: #6366f1 !important;
}
/* Sidebar inputs */
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: #ffffff !important;
    border-color: #e5e7eb !important;
    color: #0f172a !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
section[data-testid="stSidebar"] .stMultiSelect > div > div:focus-within {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #6366f1 !important;
}
/* Slider */
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #4f46e5 !important;
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 4px rgba(79,70,229,0.10) !important;
}
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] [data-baseweb="slider"] div[class*="value"] {
    color: #64748b !important;
    font-size: 0.72rem !important;
}

/* ── Scrollbar ───────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f3f4f6; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #818cf8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ── Radio buttons (Lookup-by, etc.) ─────────── */
.stRadio [role="radiogroup"] label {
    color: #0f172a !important;
}
.stRadio [role="radio"][aria-checked="true"] + div {
    color: #4f46e5 !important;
}

/* ── Slider track ────────────────────────────── */
[data-baseweb="slider"] [role="slider"] {
    background: #4f46e5 !important;
    border-color: #4f46e5 !important;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Metro BFPS — BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)

required = [
    OUTPUT_DIR / "model_dataset_selected_metros.csv",
    OUTPUT_DIR / "project_metadata.json",
    OUTPUT_DIR / "business_feasibility_scores.csv",
    OUTPUT_DIR / "full_metro_ranking.csv",
    OUTPUT_DIR / "metro_business_pivot.csv",
    OUTPUT_DIR / "business_recommendation_engine.csv",
    OUTPUT_DIR / "city_to_metro_lookup.csv",
    OUTPUT_DIR / "classifier_validation_results.csv",
    OUTPUT_DIR / "regressor_validation_results.csv",
    OUTPUT_DIR / "classifier_feature_importance.csv",
    OUTPUT_DIR / "regressor_feature_importance.csv",
    OUTPUT_DIR / "misclassified_cases.csv",
    OUTPUT_DIR / "cluster_analysis.csv",
    OUTPUT_DIR / "sensitivity_analysis.csv",
]
missing = [p.name for p in required if not p.exists()]
if missing:
    st.error("Run `python3 main.py` first. Missing: " + ", ".join(missing))
    st.stop()

outputs = load_outputs()
df = outputs["dataset"]
metadata = outputs["metadata"]
business_scores = outputs["business_scores"]
recommendation_engine = outputs["recommendation_engine"]
full_metro_ranking = outputs["full_metro_ranking"]
metro_business_pivot = outputs["metro_business_pivot"]
city_lookup = outputs["city_lookup"]
classifier_validation = outputs["classifier_validation"]
regressor_validation = outputs["regressor_validation"]
classifier_importance = outputs["classifier_importance"]
regressor_importance = outputs["regressor_importance"]
misclassified_cases = outputs["misclassified_cases"]
cv_fold_misclassified = outputs["cv_fold_misclassified"]
cv_fold_assignments = outputs["cv_fold_assignments"]
cluster_analysis = outputs["cluster_analysis"]
sensitivity_df = outputs["sensitivity_df"]
classification_report_txt = outputs["classification_report"]
regression_report_txt = outputs["regression_report"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — global filters (PowerBI-style cross-filter)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 0.4rem 0.8rem; text-align:center;">
        <div style="
            font-size:2.2rem; margin-bottom:0.5rem;
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #06b6d4 100%);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 4px 14px rgba(79,70,229,0.30));
            line-height: 1;
        ">◆</div>
        <p style="color:#0f172a; font-size:1.1rem; font-weight:800;
                  margin:0; letter-spacing:-0.02em; font-family:'Inter',sans-serif;">
            Metro BFPS
        </p>
        <p style="color:#64748b; font-size:0.65rem; margin:0.3rem 0 0;
                  text-transform:uppercase; letter-spacing:0.14em; font-weight:600;">
            Business Feasibility System
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#e5e7eb;margin:0 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#4338ca;font-size:0.68rem;font-weight:700;text-transform:uppercase;
              letter-spacing:0.12em;margin:0 0 0.4rem 0;padding-left:2px;">
        ⚙ &nbsp;Global Filters
    </p>
    """, unsafe_allow_html=True)
    st.caption("Filters apply across all dashboards.")

    region_options = sorted(df["region"].dropna().unique().tolist())
    selected_regions = st.multiselect(
        "Region", region_options, default=region_options, key="flt_region"
    )

    state_options = sorted(df["primary_state"].dropna().unique().tolist())
    selected_states = st.multiselect(
        "State", state_options, default=[], key="flt_state",
        help="Empty = all states",
    )

    pop_min, pop_max = int(df["population"].min()), int(df["population"].max())
    pop_range = st.slider(
        "Population range",
        min_value=pop_min, max_value=pop_max,
        value=(pop_min, pop_max), step=10000,
        key="flt_pop",
    )

    score_min, score_max = float(df["opportunity_score"].min()), float(df["opportunity_score"].max())
    score_range = st.slider(
        "Opportunity score range",
        min_value=round(score_min, 3), max_value=round(score_max, 3),
        value=(round(score_min, 3), round(score_max, 3)), step=0.01,
        key="flt_score",
    )

    only_high_opp = st.checkbox("High-opportunity metros only", value=False, key="flt_high")

    if st.button("↺ Reset all filters", use_container_width=True):
        for k in ["flt_region", "flt_state", "flt_pop", "flt_score", "flt_high"]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Watchlist (NEW Phase 5) ──────────────────────────────────────────────
    st.markdown('<hr style="border-color:#e5e7eb;margin:1rem 0 0.6rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#4338ca;font-size:0.68rem;font-weight:700;text-transform:uppercase;
              letter-spacing:0.12em;margin:0 0 0.4rem 0;padding-left:2px;">
        ⭐ &nbsp;Watchlist
    </p>
    """, unsafe_allow_html=True)
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    if not st.session_state.watchlist:
        st.caption("No metros saved yet. Open Market Explorer and click ⭐.")
    else:
        for _wl_metro in list(st.session_state.watchlist):
            wcol1, wcol2 = st.columns([5, 1])
            with wcol1:
                st.markdown(
                    f'<div style="font-size:0.83rem;color:#1e293b;font-weight:500;'
                    f'padding:4px 8px;background:#eef2ff;border-radius:6px;'
                    f'border:1px solid #c7d2fe;margin-bottom:4px;">'
                    f'{_wl_metro}</div>',
                    unsafe_allow_html=True,
                )
            with wcol2:
                if st.button("✕", key=f"sb_rm_{_wl_metro}", help="Remove"):
                    st.session_state.watchlist.remove(_wl_metro)
                    st.rerun()
        if st.button("Clear all", use_container_width=True, key="wl_clear"):
            st.session_state.watchlist = []
            st.rerun()

    # ── Executive Deck Download (NEW Phase 5) ────────────────────────────────
    st.markdown('<hr style="border-color:#e5e7eb;margin:1rem 0 0.6rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#4338ca;font-size:0.68rem;font-weight:700;text-transform:uppercase;
              letter-spacing:0.12em;margin:0 0 0.4rem 0;padding-left:2px;">
        📊 &nbsp;Executive Deliverables
    </p>
    """, unsafe_allow_html=True)
    _pptx = OUTPUT_DIR / "executive_summary.pptx"
    _pdf  = OUTPUT_DIR / "executive_summary.pdf"
    if _pptx.exists():
        st.download_button(
            "⬇ Executive Deck (.pptx)",
            data=_pptx.read_bytes(),
            file_name="executive_summary.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True, key="dl_exec_pptx",
        )
    if _pdf.exists():
        st.download_button(
            "⬇ Executive PDF",
            data=_pdf.read_bytes(),
            file_name="executive_summary.pdf",
            mime="application/pdf",
            use_container_width=True, key="dl_exec_pdf",
        )
    if not _pptx.exists() and not _pdf.exists():
        st.caption("Run `python3 main.py` to generate executive deliverables.")

    st.markdown('<hr style="border-color:#e5e7eb;margin:1rem 0 0.8rem 0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.78rem; line-height:1.9; padding:0 2px 1rem;
                background:#ffffff; border:1px solid #e5e7eb; border-radius:10px;
                box-shadow:0 1px 3px rgba(15,23,42,0.04);">
        <p style="color:#4338ca;font-size:0.62rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.12em;margin:0.7rem 0.8rem 0.5rem;">📦 Data coverage</p>
        <div style="padding:0 0.8rem 0.7rem;">
          <span style="color:#64748b;">Metros</span>
          <span style="float:right;color:#4f46e5;font-weight:700;">
              {metadata['selected_metros']}
          </span><br>
          <span style="color:#64748b;">Business types</span>
          <span style="float:right;color:#4f46e5;font-weight:700;">
              {metadata['business_types_supported']}
          </span><br>
          <span style="color:#64748b;">CBP rows</span>
          <span style="float:right;color:#4f46e5;font-weight:700;">
              {int(metadata['raw_cbp_rows_used']):,}
          </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Apply global filters
def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    if selected_regions:
        f = f[f["region"].isin(selected_regions)]
    if selected_states:
        f = f[f["primary_state"].isin(selected_states)]
    f = f[(f["population"] >= pop_range[0]) & (f["population"] <= pop_range[1])]
    f = f[(f["opportunity_score"] >= score_range[0]) & (f["opportunity_score"] <= score_range[1])]
    if only_high_opp and "high_opportunity" in f.columns:
        f = f[f["high_opportunity"] == 1]
    return f


df_f = apply_filters(df)
business_scores_f = business_scores.merge(
    df_f[["metro_title"]].drop_duplicates(), on="metro_title", how="inner"
)
recommendation_engine_f = recommendation_engine.merge(
    df_f[["metro_title"]].drop_duplicates(), on="metro_title", how="inner"
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="
    background:
        radial-gradient(circle at 85% 30%, rgba(99,102,241,0.55), transparent 55%),
        radial-gradient(circle at 20% 120%, rgba(6,182,212,0.35), transparent 55%),
        linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    padding: 2rem 2.3rem 1.8rem;
    border-radius: 18px;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 32px rgba(15,23,42,0.22), 0 1px 0 rgba(255,255,255,0.05) inset;
    border: 1px solid rgba(99,102,241,0.18);
">
    <!-- Subtle grid pattern overlay -->
    <div style="
        position:absolute; inset:0; opacity:0.18; pointer-events:none;
        background-image:
            linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
        background-size: 32px 32px;
    "></div>
    <h1 style="
        color:#ffffff; font-size:1.95rem; font-weight:800;
        margin:0; font-family:'Inter',sans-serif; letter-spacing:-0.03em;
        text-shadow: 0 2px 12px rgba(15,23,42,0.35);
        position: relative; z-index: 2;
    "><span style="
        background: linear-gradient(135deg, #818cf8 0%, #c7d2fe 50%, #67e8f9 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    ">◆</span>&nbsp;Metropolitan Business Feasibility</h1>
    <p style="
        color:#cbd5e1; margin:0.55rem 0 0;
        font-size:0.93rem; font-family:'Inter',sans-serif; line-height:1.6;
        position: relative; z-index: 2;
    ">
        Filtered view:&nbsp;
        <span style="color:#a5b4fc; font-weight:700;">{len(df_f):,} of {len(df):,} metros</span>
        &nbsp;·&nbsp; CBP 2022 + ACS 2024 + BLS 2024
        &nbsp;·&nbsp;
        <span style="color:#67e8f9; font-weight:600;">
            {int(df_f['high_opportunity'].sum())} high-opportunity
        </span>
    </p>
</div>
""", unsafe_allow_html=True)

if df_f.empty:
    st.warning("No metros match the current filters. Adjust filters in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🏆 Executive Overview",
    "🏙️ Market Explorer",
    "💼 Business Opportunity",
    "🧪 What-If Scenario Lab",
    "🔄 Compare Metros",
    "🗂️ Cluster Analysis",
    "📐 Sensitivity Analysis",
    "🤖 Model Insights",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.8rem;">Key Performance Indicators</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Metros (filtered)", f"{len(df_f):,}")
    c2.metric("High-Opportunity", f"{int(df_f['high_opportunity'].sum()):,}",
              delta=f"{df_f['high_opportunity'].mean()*100:.0f}% of filtered")
    c3.metric("Avg Median Income", f"${df_f['median_income'].mean():,.0f}")
    c4.metric("Avg Unemployment", f"{df_f['unemployment_rate'].mean():.2f}%")
    c5.metric("Total Establishments", f"{int(df_f['est'].sum()):,}")

    st.markdown("### 🗺️ Opportunity Score by State (Choropleth)")
    state_avg = (
        df_f.groupby("primary_state", as_index=False)
        .agg(opportunity_score=("opportunity_score", "mean"),
             metros=("metro_title", "count"),
             median_income=("median_income", "mean"),
             unemployment=("unemployment_rate", "mean"))
    )
    fig_map = px.choropleth(
        state_avg,
        locations="primary_state",
        locationmode="USA-states",
        color="opportunity_score",
        scope="usa",
        color_continuous_scale=DESERT_SCALE,
        hover_data={"metros": True, "median_income": ":,.0f", "unemployment": ":.2f"},
        labels={"opportunity_score": "Avg Opp. Score"},
    )
    clean_layout(fig_map, height=480)
    st.plotly_chart(fig_map, use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### 🏆 Top 15 Metros by Opportunity Score")
        top15 = df_f.nlargest(15, "opportunity_score")[
            ["metro_title", "opportunity_score", "region", "median_income", "unemployment_rate"]
        ]
        fig_top = px.bar(
            top15.iloc[::-1],
            x="opportunity_score", y="metro_title", orientation="h",
            color="region", color_discrete_map=REGION_PALETTE,
            hover_data={"median_income": ":,.0f", "unemployment_rate": ":.2f"},
            labels={"opportunity_score": "Opportunity Score", "metro_title": ""},
        )
        clean_layout(fig_top, height=520)
        st.plotly_chart(fig_top, use_container_width=True)

    with right:
        st.markdown("### 🌎 Region Mix")
        region_counts = df_f["region"].value_counts().reset_index()
        region_counts.columns = ["region", "metros"]
        fig_pie = px.pie(
            region_counts, names="region", values="metros", hole=0.5,
            color="region", color_discrete_map=REGION_PALETTE,
        )
        fig_pie.update_traces(textinfo="label+percent")
        clean_layout(fig_pie, height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Score Distribution")
        fig_hist = px.histogram(
            df_f, x="opportunity_score", nbins=25, color="region",
            color_discrete_map=REGION_PALETTE,
        )
        clean_layout(fig_hist, height=260)
        st.plotly_chart(fig_hist, use_container_width=True)

    # NEW — Lat/lon bubble map (Phase 5)
    if "lat" in df_f.columns and df_f["lat"].notna().any():
        st.markdown("### 🗺️ Metro Bubble Map (size = population, color = opportunity score)")
        st.caption(
            "Geographic distribution of metros. Hover for details, click and drag to pan."
        )
        bubble = df_f.dropna(subset=["lat", "lon"]).copy()
        fig_bubble = px.scatter_geo(
            bubble,
            lat="lat", lon="lon",
            size="population", color="opportunity_score",
            color_continuous_scale=DESERT_SCALE,
            hover_name="metro_title",
            hover_data={"region": True, "primary_state": True,
                        "median_income": ":,.0f",
                        "unemployment_rate": ":.2f",
                        "opportunity_score": ":.3f",
                        "population": ":,",
                        "lat": False, "lon": False},
            scope="usa",
            size_max=42,
            labels={"opportunity_score": "Opp. Score"},
        )
        fig_bubble.update_geos(
            showland=True, landcolor="#f8fafc",
            showsubunits=True, subunitcolor="#cbd5e1",
            showcountries=True, countrycolor="#94a3b8",
            showlakes=True, lakecolor="#e0f2fe",
            bgcolor="rgba(0,0,0,0)",
        )
        fig_bubble.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=520,
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("### 💰 Income vs Unemployment (size = population)")
    fig_sc = px.scatter(
        df_f, x="median_income", y="unemployment_rate",
        size="population", color="opportunity_score",
        color_continuous_scale=DESERT_SCALE,
        hover_name="metro_title",
        hover_data={"region": True, "population": ":,",
                    "median_income": ":,.0f", "unemployment_rate": ":.2f",
                    "opportunity_score": ":.3f"},
        labels={"median_income": "Median Household Income",
                "unemployment_rate": "Unemployment Rate (%)"},
        size_max=45,
    )
    clean_layout(fig_sc, height=520)
    st.plotly_chart(fig_sc, use_container_width=True)

    # Cost-of-Living impact section (NEW — BEA RPP)
    if "cost_of_living_index" in df_f.columns and df_f["cost_of_living_index"].notna().any():
        st.markdown("### 🏠 Cost of Living Impact — Real vs Gross Income")
        st.caption(
            "BEA Regional Price Parities (100 = US average). Metros above the diagonal have "
            "higher purchasing power than gross income suggests; metros below are cost-burdened."
        )
        ci1, ci2 = st.columns([1.4, 1])
        with ci1:
            fig_real = px.scatter(
                df_f, x="median_income", y="real_median_income",
                size="population", color="cost_of_living_index",
                color_continuous_scale=DESERT_DIVERGING_R,
                hover_name="metro_title",
                hover_data={"region": True, "population": ":,",
                            "cost_of_living_index": ":.1f",
                            "housing_cost_index": ":.1f",
                            "median_income": ":,.0f",
                            "real_median_income": ":,.0f"},
                labels={"median_income": "Gross Median Income ($)",
                        "real_median_income": "Real (PPP-adjusted) Income ($)",
                        "cost_of_living_index": "CoL Index"},
                size_max=40,
            )
            x_lo = float(df_f["median_income"].min())
            x_hi = float(df_f["median_income"].max())
            fig_real.add_shape(type="line", x0=x_lo, x1=x_hi, y0=x_lo, y1=x_hi,
                               line=dict(color="#cbd5e1", width=2, dash="dash"))
            fig_real.add_annotation(
                x=x_hi, y=x_hi, text="Gross = Real (US avg cost)",
                showarrow=False, font=dict(size=10, color="#94a3b8"),
                xanchor="right", yanchor="bottom",
            )
            clean_layout(fig_real, height=480)
            st.plotly_chart(fig_real, use_container_width=True)
        with ci2:
            st.markdown("**Most affordable metros**")
            cheapest = df_f.nsmallest(8, "cost_of_living_index")[
                ["metro_title", "primary_state", "cost_of_living_index", "real_median_income"]
            ]
            st.dataframe(
                cheapest.rename(columns={
                    "metro_title": "Metro", "primary_state": "ST",
                    "cost_of_living_index": "CoL", "real_median_income": "Real Inc.",
                }).style.format({"CoL": "{:.1f}", "Real Inc.": "${:,.0f}"}),
                use_container_width=True, hide_index=True,
            )
            st.markdown("**Most expensive metros**")
            priciest = df_f.nlargest(8, "cost_of_living_index")[
                ["metro_title", "primary_state", "cost_of_living_index", "real_median_income"]
            ]
            st.dataframe(
                priciest.rename(columns={
                    "metro_title": "Metro", "primary_state": "ST",
                    "cost_of_living_index": "CoL", "real_median_income": "Real Inc.",
                }).style.format({"CoL": "{:.1f}", "Real Inc.": "${:,.0f}"}),
                use_container_width=True, hide_index=True,
            )

    with st.expander("📥 Download filtered data"):
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇ Filtered metro ranking (CSV)",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name="filtered_metro_ranking.csv", mime="text/csv",
        )
        c2.download_button(
            "⬇ Full metro ranking (CSV)",
            data=full_metro_ranking.to_csv(index=False).encode("utf-8"),
            file_name="full_metro_ranking.csv", mime="text/csv",
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.8rem;">Single Metro Deep-Dive</p>', unsafe_allow_html=True)

    lookup_mode = st.radio("Lookup by:", ["Metro name", "City alias"], horizontal=True, key="me_mode")
    if lookup_mode == "City alias":
        alias = st.selectbox("City alias", sorted(city_lookup["city_alias"].unique()))
        mapped = city_lookup[city_lookup["city_alias"] == alias]
        metro_name = st.selectbox("Mapped metro", mapped["metro_title"].tolist())
    else:
        metro_name = st.selectbox(
            "Metro market",
            sorted(df["metro_title"].unique()),
            index=0,
        )

    row = df[df["metro_title"] == metro_name].iloc[0]

    # NEW (Phase 5) — Watchlist + Dossier action bar
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    is_saved = metro_name in st.session_state.watchlist
    act_a, act_b, act_c = st.columns([1, 1, 4])
    with act_a:
        if not is_saved:
            if st.button("⭐ Add to watchlist", key=f"add_{metro_name}", use_container_width=True):
                st.session_state.watchlist.append(metro_name)
                st.rerun()
        else:
            if st.button("✓ Remove from watchlist", key=f"rm_{metro_name}", use_container_width=True):
                st.session_state.watchlist.remove(metro_name)
                st.rerun()
    with act_b:
        # Per-metro dossier export (PowerPoint via exec_report.build_metro_pptx)
        try:
            import io
            from exec_report import build_metro_pptx
            if st.button("📄 Generate metro dossier", key=f"dossier_{metro_name}", use_container_width=True):
                with st.spinner("Building PowerPoint dossier..."):
                    out_path = build_metro_pptx(metro_name)
                with open(out_path, "rb") as f:
                    st.session_state[f"dossier_bytes_{metro_name}"] = f.read()
                st.session_state[f"dossier_name_{metro_name}"] = out_path.name
                st.success(f"Generated: {out_path.name}")
        except ImportError:
            st.caption("📄 Dossier export unavailable (install python-pptx)")
    with act_c:
        if f"dossier_bytes_{metro_name}" in st.session_state:
            st.download_button(
                "⬇ Download metro dossier (.pptx)",
                data=st.session_state[f"dossier_bytes_{metro_name}"],
                file_name=st.session_state[f"dossier_name_{metro_name}"],
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
                key=f"dl_{metro_name}",
            )

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Opportunity Rank", f"#{int(row['opportunity_rank'])} / {len(df)}")
    k2.metric("Opportunity Score", f"{row['opportunity_score']:.3f}")
    k3.metric("Median Income", f"${int(row['median_income']):,}")
    k4.metric("Unemployment", f"{row['unemployment_rate']:.2f}%",
              delta=f"{row['recent_unemployment_change']:+.1f} pts vs 2023",
              delta_color="inverse")
    k5.metric("Class", "🟢 High" if row["high_opportunity_metro"] else "⚪ Low/Med")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Population", f"{int(row['population']):,}")
    k2.metric("Establishments", f"{int(row['est']):,}")
    k3.metric("Employment", f"{int(row['emp']):,}")
    k4.metric("Est / 10k", f"{row['establishments_per_10k']:.1f}")
    k5.metric("Region", str(row["region"]))

    # Cost-of-Living strip (NEW — BEA RPP)
    if "cost_of_living_index" in row.index:
        col_idx = float(row["cost_of_living_index"])
        hou_idx = float(row.get("housing_cost_index", 100.0))
        real_inc = float(row.get("real_median_income", row["median_income"]))
        col_delta = col_idx - 100.0
        hou_delta = hou_idx - 100.0
        real_delta = real_inc - float(row["median_income"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Cost of Living (BEA)",
            f"{col_idx:.1f}",
            delta=f"{col_delta:+.1f} vs US avg",
            delta_color="inverse",
            help="100 = US national average. Lower = cheaper.",
        )
        c2.metric(
            "Housing Cost (BEA)",
            f"{hou_idx:.1f}",
            delta=f"{hou_delta:+.1f} vs US avg",
            delta_color="inverse",
            help="BEA RPP — Services: Housing component. 100 = US average.",
        )
        c3.metric(
            "Real Median Income",
            f"${int(real_inc):,}",
            delta=f"{real_delta:+,.0f} vs gross",
            help="Median income adjusted for local cost of living (purchasing-power-adjusted).",
        )
        if col_idx <= 92:
            verdict, vcolor = "Affordable", "#10b981"
        elif col_idx <= 102:
            verdict, vcolor = "Average cost", "#f59e0b"
        elif col_idx <= 112:
            verdict, vcolor = "High cost", "#f97316"
        else:
            verdict, vcolor = "Very high cost", "#ef4444"
        c4.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;
                        padding:1.1rem 1.3rem;height:100%;position:relative;overflow:hidden;
                        box-shadow:0 2px 8px rgba(15,23,42,0.05);">
                <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{vcolor};"></div>
                <p style="font-size:0.7rem;font-weight:600;color:#64748b;
                          text-transform:uppercase;letter-spacing:0.07em;margin:0 0 6px;">
                    Affordability Verdict
                </p>
                <p style="font-size:1.2rem;font-weight:700;color:{vcolor};margin:0;line-height:1.2;">
                    {verdict}
                </p>
                <p style="font-size:0.72rem;color:#64748b;margin:4px 0 0;">
                    Index {col_idx:.0f} {'·' if col_idx != 100 else ''}
                    {('+' if col_delta>=0 else '')}{col_delta:.0f} vs US
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Gauge for opportunity score
    pct = float((df["opportunity_score"] < row["opportunity_score"]).mean() * 100)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=row["opportunity_score"],
        title={"text": f"Opportunity Score Percentile: {pct:.0f}%"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": SAGE_500},
            "steps": [
                {"range": [0, 0.3], "color": "#fee2e2"},
                {"range": [0.3, 0.6], "color": "#fef3c7"},
                {"range": [0.6, 1.0], "color": "#d1fae5"},
            ],
            "threshold": {"line": {"color": SAND_800, "width": 4},
                          "thickness": 0.75, "value": float(df["opportunity_score"].quantile(0.7))},
        },
    ))
    clean_layout(fig_gauge, height=280)

    g_left, g_right = st.columns([1, 1.4])
    with g_left:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with g_right:
        st.markdown("### 💼 All 6 Business Feasibility Scores for this Metro")
        biz = (
            business_scores[business_scores["metro_title"] == metro_name]
            [["business_type", "business_feasibility_score", "feasibility_band",
              "entry_strategy", "market_strength_probability", "category_fit_score"]]
            .sort_values("business_feasibility_score", ascending=True)
        )
        fig_biz = px.bar(
            biz, x="business_feasibility_score", y="business_type",
            orientation="h", color="entry_strategy",
            color_discrete_map=STRATEGY_PALETTE,
            hover_data={"feasibility_band": True,
                        "market_strength_probability": ":.3f",
                        "category_fit_score": ":.3f"},
            labels={"business_feasibility_score": "Feasibility Score (/100)",
                    "business_type": ""},
        )
        clean_layout(fig_biz, height=320)
        st.plotly_chart(fig_biz, use_container_width=True)

    # Sector employment shares
    st.markdown("### 🏭 Sector Employment Composition")
    sector_rows = []
    for col in [c for c in df.columns if c.startswith("emp_share_")]:
        code = col.split("_")[-1]
        sector_rows.append({"sector": NAICS_LABELS.get(code, code),
                            "share": float(row[col])})
    sec = pd.DataFrame(sector_rows).query("share > 0").sort_values("share", ascending=True)
    fig_sec = px.bar(
        sec, x="share", y="sector", orientation="h",
        color="share", color_continuous_scale=DESERT_SCALE,
        labels={"share": "Employment Share", "sector": ""},
    )
    fig_sec.update_xaxes(tickformat=".1%")
    clean_layout(fig_sec, height=520)
    st.plotly_chart(fig_sec, use_container_width=True)

    # Unemployment trend 2020-2024 for selected metro
    trend_cols_app = ["unemployment_2020", "unemployment_2021", "unemployment_2022",
                      "unemployment_2023", "unemployment_rate"]
    trend_years_app = [2020, 2021, 2022, 2023, 2024]
    available_trend = [c for c in trend_cols_app if c in df.columns]
    if len(available_trend) >= 2:
        st.markdown("### 📉 Unemployment Trend 2020–2024")
        trend_vals = [float(row[c]) if c in row.index else None for c in available_trend]
        trend_df_app = pd.DataFrame({"Year": trend_years_app[:len(available_trend)],
                                     "Unemployment Rate (%)": trend_vals})
        fig_trend = px.line(trend_df_app, x="Year", y="Unemployment Rate (%)",
                            markers=True, color_discrete_sequence=[TERRA_500])
        fig_trend.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BUSINESS OPPORTUNITY
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.8rem;">Per-Business Opportunity Analysis</p>', unsafe_allow_html=True)

    biz_type = st.selectbox(
        "Business type",
        sorted(business_scores["business_type"].unique()),
        key="bo_type",
    )
    sub = business_scores_f[business_scores_f["business_type"] == biz_type].copy()
    if sub.empty:
        st.info("No metros match the global filters for this business type.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Feasibility", f"{sub['business_feasibility_score'].mean():.1f}/100")
    k2.metric("Top Score", f"{sub['business_feasibility_score'].max():.1f}/100")
    k3.metric("Established Leaders",
              int((sub["entry_strategy"] == "Established Leader").sum()))
    k4.metric("White-Space Markets",
              int((sub["entry_strategy"] == "White-Space Opportunity").sum()))

    # Choropleth — average feasibility by state for selected business
    state_biz = sub.groupby("primary_state", as_index=False)["business_feasibility_score"].mean()
    fig_st = px.choropleth(
        state_biz, locations="primary_state", locationmode="USA-states",
        color="business_feasibility_score", scope="usa",
        color_continuous_scale=DESERT_SCALE,
        labels={"business_feasibility_score": "Avg Score"},
    )
    clean_layout(fig_st, height=460)
    st.markdown(f"### 🗺️ Average {biz_type} Feasibility by State")
    st.plotly_chart(fig_st, use_container_width=True)

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown(f"### 🏆 Top 20 Metros — {biz_type}")
        top20 = sub.nlargest(20, "business_feasibility_score")
        fig_b = px.bar(
            top20.iloc[::-1],
            x="business_feasibility_score", y="metro_title", orientation="h",
            color="entry_strategy", color_discrete_map=STRATEGY_PALETTE,
            hover_data={"market_strength_probability": ":.3f",
                        "category_fit_score": ":.3f",
                        "feasibility_band": True},
            labels={"business_feasibility_score": "Feasibility Score",
                    "metro_title": ""},
        )
        clean_layout(fig_b, height=620)
        st.plotly_chart(fig_b, use_container_width=True)

    with right:
        st.markdown("### 🎯 Strategy Mix")
        strat = sub["entry_strategy"].value_counts().reset_index()
        strat.columns = ["entry_strategy", "metros"]
        fig_s = px.pie(strat, names="entry_strategy", values="metros", hole=0.5,
                       color="entry_strategy", color_discrete_map=STRATEGY_PALETTE)
        fig_s.update_traces(textinfo="percent+value")
        clean_layout(fig_s, height=320)
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("### Region × Strategy Heatmap")
        heat = sub.pivot_table(
            index="region", columns="entry_strategy",
            values="metro_title", aggfunc="count", fill_value=0,
        )
        fig_h = px.imshow(
            heat, text_auto=True, color_continuous_scale=DESERT_SCALE,
            aspect="auto", labels={"color": "Metros"},
        )
        clean_layout(fig_h, height=300)
        st.plotly_chart(fig_h, use_container_width=True)

    # Scatter: market strength vs category fit, with strategy color
    st.markdown(f"### 📊 Market Strength vs Category Fit — {biz_type}")
    fig_sc2 = px.scatter(
        sub, x="market_strength_probability", y="category_fit_score",
        size="population", color="entry_strategy",
        color_discrete_map=STRATEGY_PALETTE,
        hover_name="metro_title",
        hover_data={"business_feasibility_score": ":.1f",
                    "median_income": ":,.0f"},
        labels={"market_strength_probability": "Modeled Market Strength",
                "category_fit_score": "Category Fit Score"},
        size_max=40,
    )
    clean_layout(fig_sc2, height=520)
    st.plotly_chart(fig_sc2, use_container_width=True)

    with st.expander("📥 Download data"):
        st.download_button(
            f"⬇ {biz_type} feasibility (CSV)",
            data=sub.to_csv(index=False).encode("utf-8"),
            file_name=f"feasibility_{biz_type.lower().replace(' ', '_').replace('/', '_')}.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇ All business feasibility scores (CSV)",
            data=business_scores.to_csv(index=False).encode("utf-8"),
            file_name="business_feasibility_scores.csv", mime="text/csv",
            key="dl_all_biz",
        )

    # ── Smart Recommendations ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p style="font-size:0.78rem;font-weight:700;color:#4338ca;text-transform:uppercase;letter-spacing:0.12em;margin:1rem 0 0.5rem;">🎯 Smart Recommendations</p>', unsafe_allow_html=True)
    st.caption("Tell us your priorities — we'll rank the best markets for you.")

    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        rec_biz = st.selectbox("My business type", sorted(business_scores["business_type"].unique()),
                               key="rec_smart_biz")
    with rec_col2:
        priority = st.selectbox(
            "I care most about",
            ["Highest feasibility score", "Lowest competition (white-space)",
             "Fastest improving labor market", "Established proven market"],
            key="rec_priority",
        )
    with rec_col3:
        rec_region = st.selectbox("Preferred region", ["Any"] + sorted(df["region"].dropna().unique()),
                                  key="rec_region_filter")

    rec_sub = business_scores_f[business_scores_f["business_type"] == rec_biz].copy()
    if rec_region != "Any":
        rec_sub = rec_sub[rec_sub["region"] == rec_region]

    sort_col_map = {
        "Highest feasibility score": ("business_feasibility_score", False),
        "Lowest competition (white-space)": ("whitespace_opportunity_score", False),
        "Fastest improving labor market": ("momentum_score", False),
        "Established proven market": ("business_feasibility_score", False),
    }
    strategy_filter_map = {
        "Highest feasibility score": None,
        "Lowest competition (white-space)": "White-Space Opportunity",
        "Fastest improving labor market": "Momentum Market",
        "Established proven market": "Established Leader",
    }
    sort_col, sort_asc = sort_col_map[priority]
    strat_filter = strategy_filter_map[priority]
    if strat_filter:
        filtered_rec = rec_sub[rec_sub["entry_strategy"] == strat_filter]
        if filtered_rec.empty:
            filtered_rec = rec_sub
    else:
        filtered_rec = rec_sub

    top_picks = filtered_rec.nlargest(5, sort_col)

    rec_cols = st.columns(min(5, len(top_picks)))
    for i, (_, pick) in enumerate(top_picks.iterrows()):
        with rec_cols[i]:
            st.markdown(f"**#{i+1}** {pick['metro_title']}")
            st.metric("Score", f"{pick['business_feasibility_score']:.1f}")
            st.caption(f"{pick['entry_strategy']}\n{pick['region']}, {pick['primary_state']}")

    fig_rec = px.bar(
        top_picks.iloc[::-1], x=sort_col, y="metro_title",
        orientation="h", color="entry_strategy",
        color_discrete_map=STRATEGY_PALETTE,
        text=top_picks.iloc[::-1][sort_col].round(1),
        labels={sort_col: sort_col.replace("_", " ").title(), "metro_title": ""},
    )
    clean_layout(fig_rec, height=340)
    st.plotly_chart(fig_rec, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — WHAT-IF SCENARIO LAB
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.8rem;">Live What-If Scenario Simulator</p>', unsafe_allow_html=True)
    st.caption(
        "Adjust market and category levers — every chart updates in real time. "
        "The simulator uses the transparent score components from the model."
    )

    sc_left, sc_right = st.columns([1, 1])
    with sc_left:
        sc_biz = st.selectbox("Business type", sorted(business_scores["business_type"].unique()),
                              key="sc_biz")
    with sc_right:
        sc_metro = st.selectbox("Metro market", sorted(df["metro_title"].unique()),
                                key="sc_metro")

    sub_full = business_scores[business_scores["business_type"] == sc_biz]
    sub_row = sub_full[sub_full["metro_title"] == sc_metro].iloc[0]

    st.markdown("#### 🎚️ Levers")
    s1, s2, s3, s4, s5 = st.columns(5)
    income_pct = s1.slider("Income Δ (%)", -25, 25, 0, 1)
    unemp_delta = s2.slider("Unemployment Δ (pts)", -3.0, 3.0, 0.0, 0.1)
    payroll_pct = s3.slider("Payroll/employee Δ (%)", -25, 25, 0, 1)
    sec_share_pct = s4.slider("Category share Δ (%)", -50, 50, 0, 1)
    sec_dens_pct = s5.slider("Category density Δ (%)", -50, 50, 0, 1)

    # Compute baseline + scenario decompositions
    base_inc = float(sub_row["median_income"])
    base_unemp = float(sub_row["unemployment_rate"])
    base_pay = float(sub_row["annual_payroll_per_employee"])
    base_share = float(sub_row["sector_employment_share"])
    base_dens = float(sub_row["sector_establishments_per_10k"])
    base_est10k = float(sub_row["establishments_per_10k"])
    base_ent = float(sub_row["industry_entropy"])

    new_inc = base_inc * (1 + income_pct / 100.0)
    new_unemp = max(0.0, base_unemp + unemp_delta)
    new_pay = base_pay * (1 + payroll_pct / 100.0)
    new_share = base_share * (1 + sec_share_pct / 100.0)
    new_dens = base_dens * (1 + sec_dens_pct / 100.0)

    def market_strength(inc, unemp, pay):
        return (
            0.25 * scale_value(inc, business_scores["median_income"])
            + 0.25 * scale_value(business_scores["unemployment_rate"].max() - unemp,
                                 business_scores["unemployment_rate"].max() - business_scores["unemployment_rate"])
            + 0.20 * scale_value(pay, business_scores["annual_payroll_per_employee"])
            + 0.15 * scale_value(base_est10k, business_scores["establishments_per_10k"])
            + 0.15 * scale_value(base_ent, business_scores["industry_entropy"])
        )

    def category_fit(share, dens):
        return (
            0.60 * scale_value(share, sub_full["sector_employment_share"])
            + 0.40 * scale_value(dens, sub_full["sector_establishments_per_10k"])
        )

    base_market = market_strength(base_inc, base_unemp, base_pay)
    sc_market = market_strength(new_inc, new_unemp, new_pay)
    base_fit = category_fit(base_share, base_dens)
    sc_fit = category_fit(new_share, new_dens)

    base_score = 100 * (0.70 * base_market + 0.30 * base_fit)
    sc_score = 100 * (0.70 * sc_market + 0.30 * sc_fit)
    delta = sc_score - base_score

    # Top KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline Score", f"{base_score:.1f}/100")
    k2.metric("Scenario Score", f"{sc_score:.1f}/100", delta=f"{delta:+.1f}")
    k3.metric("Market Strength Δ", f"{(sc_market - base_market):+.3f}")
    k4.metric("Category Fit Δ", f"{(sc_fit - base_fit):+.3f}")

    # Side-by-side gauges
    g1, g2 = st.columns(2)
    with g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=base_score,
            title={"text": "Baseline"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": SAND_400},
                   "steps": [{"range": [0, 40], "color": "#fee2e2"},
                             {"range": [40, 70], "color": "#fef3c7"},
                             {"range": [70, 100], "color": "#d1fae5"}]},
        ))
        clean_layout(fig_g1, height=280)
        st.plotly_chart(fig_g1, use_container_width=True)
    with g2:
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=sc_score,
            delta={"reference": base_score, "increasing": {"color": SAGE_500},
                   "decreasing": {"color": "#b22222"}},
            title={"text": "Scenario"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": SAGE_500},
                   "steps": [{"range": [0, 40], "color": "#fee2e2"},
                             {"range": [40, 70], "color": "#fef3c7"},
                             {"range": [70, 100], "color": "#d1fae5"}]},
        ))
        clean_layout(fig_g2, height=280)
        st.plotly_chart(fig_g2, use_container_width=True)

    # Component decomposition
    comp_df = pd.DataFrame({
        "Component": ["Market Strength (70%)", "Category Fit (30%)"],
        "Baseline": [70 * base_market, 30 * base_fit],
        "Scenario": [70 * sc_market, 30 * sc_fit],
    }).melt(id_vars="Component", var_name="Variant", value_name="Contribution")

    fig_comp = px.bar(
        comp_df, x="Component", y="Contribution", color="Variant",
        barmode="group", text=comp_df["Contribution"].round(1),
        color_discrete_map={"Baseline": SAND_400, "Scenario": SAGE_500},
    )
    clean_layout(fig_comp, height=380)
    st.markdown("#### 🔬 Score Decomposition (out of 100)")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Where would this metro rank now?
    base_rank = int((sub_full["business_feasibility_score"] > base_score).sum() + 1)
    sc_rank_estimate = int((sub_full["business_feasibility_score"] > sc_score).sum() + 1)
    rk1, rk2, rk3 = st.columns(3)
    rk1.metric(f"Baseline rank in {sc_biz}", f"#{base_rank} / {len(sub_full)}")
    rk2.metric(f"Scenario rank in {sc_biz}", f"#{sc_rank_estimate} / {len(sub_full)}",
               delta=f"{base_rank - sc_rank_estimate:+d} positions")
    rk3.metric("Adjusted income", f"${int(new_inc):,}",
               delta=f"{income_pct:+d}%")

    st.markdown("#### 📋 Adjusted Inputs")
    adj = pd.DataFrame({
        "Lever": ["Median income", "Unemployment", "Payroll/employee",
                 "Category employment share", "Category establishments per 10k"],
        "Baseline": [f"${base_inc:,.0f}", f"{base_unemp:.2f}%",
                     f"${base_pay:,.0f}", f"{base_share:.1%}", f"{base_dens:.2f}"],
        "Scenario": [f"${new_inc:,.0f}", f"{new_unemp:.2f}%",
                     f"${new_pay:,.0f}", f"{new_share:.1%}", f"{new_dens:.2f}"],
    })
    st.dataframe(adj, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARE METROS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.8rem;">Side-by-Side Metro Comparison</p>', unsafe_allow_html=True)
    st.caption("Pick 2–4 metros to benchmark them across every dimension.")

    default_top = df.nlargest(3, "opportunity_score")["metro_title"].tolist()
    chosen = st.multiselect(
        "Compare metros (pick 2–4)",
        sorted(df["metro_title"].unique()),
        default=default_top,
        max_selections=4,
    )

    if len(chosen) < 2:
        st.info("Select at least 2 metros above to start the comparison.")
        st.stop()

    cmp = df[df["metro_title"].isin(chosen)].copy()

    # KPI strip across metros
    cols = st.columns(len(chosen))
    for i, m in enumerate(chosen):
        r = cmp[cmp["metro_title"] == m].iloc[0]
        with cols[i]:
            st.markdown(f"#### {m}")
            st.metric("Opportunity Rank", f"#{int(r['opportunity_rank'])}")
            st.metric("Score", f"{r['opportunity_score']:.3f}")
            st.metric("Income", f"${int(r['median_income']):,}")
            st.metric("Unemployment", f"{r['unemployment_rate']:.2f}%")
            st.metric("Population", f"{int(r['population']):,}")

    # Radar chart — opportunity score components
    components = ["median_income_norm", "inverse_unemployment_norm",
                  "annual_payroll_per_employee_norm", "establishments_per_10k_norm",
                  "industry_entropy_norm"]
    component_labels = ["Income", "Low Unemp.", "Payroll/emp",
                        "Bus. Density", "Diversity"]
    fig_r = go.Figure()
    palette = [SAGE_500, TERRA_500, SAND_500, SAND_300]
    for i, m in enumerate(chosen):
        r = cmp[cmp["metro_title"] == m].iloc[0]
        fig_r.add_trace(go.Scatterpolar(
            r=[float(r[c]) for c in components],
            theta=component_labels,
            fill="toself", name=m,
            line_color=palette[i % len(palette)],
        ))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=500,
    )
    st.markdown("### 📐 Opportunity Score Component Radar")
    st.plotly_chart(fig_r, use_container_width=True)

    # Business feasibility comparison
    st.markdown("### 💼 Business Feasibility Score by Type")
    cmp_biz = business_scores[business_scores["metro_title"].isin(chosen)][
        ["metro_title", "business_type", "business_feasibility_score"]
    ]
    fig_cb = px.bar(
        cmp_biz, x="business_type", y="business_feasibility_score",
        color="metro_title", barmode="group",
        labels={"business_feasibility_score": "Feasibility (/100)",
                "business_type": ""},
    )
    fig_cb.update_xaxes(tickangle=20)
    clean_layout(fig_cb, height=460)
    st.plotly_chart(fig_cb, use_container_width=True)

    # Detailed comparison table
    st.markdown("### 📋 Full Comparison Table")
    detail_cols = ["metro_title", "primary_state", "region", "opportunity_score",
                   "opportunity_rank", "median_income", "unemployment_rate",
                   "recent_unemployment_change", "population", "est",
                   "establishments_per_10k", "annual_payroll_per_employee",
                   "industry_entropy", "high_opportunity_metro"]
    detail = cmp[detail_cols].set_index("metro_title").T
    st.dataframe(detail, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — CLUSTER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.8rem;">Metro Cluster Analysis — K-Means (k=5)</p>', unsafe_allow_html=True)
    st.caption(
        "Metros are grouped by their structural and economic characteristics. "
        "Each cluster reveals a distinct market archetype."
    )

    # Filter cluster_analysis by the global filters (join on metro_title)
    cluster_f = cluster_analysis[cluster_analysis["metro_title"].isin(df_f["metro_title"])].copy()

    # KPI strip per cluster
    cluster_summary = cluster_f.groupby("cluster_name").agg(
        metros=("metro_title", "count"),
        avg_opp=("opportunity_score", "mean"),
        avg_income=("median_income", "mean"),
        avg_unemp=("unemployment_rate", "mean"),
    ).reset_index().sort_values("avg_opp", ascending=False)

    kpi_cols = st.columns(len(cluster_summary))
    for i, row_c in cluster_summary.iterrows():
        with kpi_cols[int(i) % len(kpi_cols)]:
            st.metric(row_c["cluster_name"], f"{int(row_c['metros'])} metros")
            st.caption(f"Opp: {row_c['avg_opp']:.3f} | Income: ${row_c['avg_income']:,.0f} | Unemp: {row_c['avg_unemp']:.1f}%")

    # PCA scatter
    st.markdown("### 🔵 Cluster Map (PCA 2-D projection)")
    cluster_palette_app = {n: c for n, c in zip(
        cluster_f["cluster_name"].unique(),
        [SAGE_500, TERRA_500, SAND_500, SAND_300, TERRA_600, SAND_400]
    )}
    fig_cl = px.scatter(
        cluster_f, x="pca_1", y="pca_2", color="cluster_name",
        color_discrete_map=cluster_palette_app,
        hover_name="metro_title",
        hover_data={"opportunity_score": ":.3f", "median_income": ":,.0f",
                    "unemployment_rate": ":.2f", "region": True},
        labels={"pca_1": "PCA Component 1", "pca_2": "PCA Component 2", "cluster_name": "Cluster"},
        size_max=10,
    )
    clean_layout(fig_cl, height=520)
    st.plotly_chart(fig_cl, use_container_width=True)

    # Cluster profile radar
    st.markdown("### 📐 Cluster Average Profiles")
    profile_cols = ["opportunity_score", "median_income", "unemployment_rate",
                    "establishments_per_10k", "annual_payroll_per_employee", "industry_entropy"]
    profile_labels = ["Opp Score", "Median Income", "Unemployment",
                      "Bus. Density", "Payroll/emp", "Diversity"]
    cluster_avg = cluster_f.groupby("cluster_name")[profile_cols].mean()
    # Normalize each column 0-1 for radar
    cluster_norm = (cluster_avg - cluster_avg.min()) / (cluster_avg.max() - cluster_avg.min()).replace(0, 1)

    fig_rad = go.Figure()
    for cname in cluster_norm.index:
        vals = list(cluster_norm.loc[cname].values) + [cluster_norm.loc[cname].values[0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals, theta=profile_labels + [profile_labels[0]],
            fill="toself", name=cname,
        ))
    fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          showlegend=True, height=500)
    st.plotly_chart(fig_rad, use_container_width=True)

    # Cluster detail table
    st.markdown("### 📋 Cluster Membership")
    cluster_detail = cluster_f[["metro_title", "cluster_name", "region", "primary_state",
                                 "opportunity_score", "median_income", "unemployment_rate",
                                 "population"]].sort_values(["cluster_name", "opportunity_score"],
                                                            ascending=[True, False])
    selected_cl = st.selectbox("View metros in cluster",
                                ["All"] + sorted(cluster_f["cluster_name"].unique()), key="cl_filter")
    if selected_cl != "All":
        cluster_detail = cluster_detail[cluster_detail["cluster_name"] == selected_cl]
    st.dataframe(cluster_detail, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download cluster analysis (CSV)",
        data=cluster_analysis.to_csv(index=False).encode("utf-8"),
        file_name="cluster_analysis.csv", mime="text/csv",
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — SENSITIVITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.8rem;">Opportunity Score Sensitivity Analysis</p>', unsafe_allow_html=True)
    st.caption(
        "Five alternative weight schemes are applied to the same data. "
        "Stable metros rank well regardless of which metric you emphasize. "
        "Volatile metros are highly dependent on which factors you prioritize."
    )

    scenario_value_cols = [c for c in sensitivity_df.columns
                           if c not in ("metro_title", "rank_mean", "rank_std", "stability")]

    k1, k2, k3 = st.columns(3)
    k1.metric("Stable metros (rank std ≤ 15)",
              int((sensitivity_df["rank_std"] <= 15).sum()))
    k2.metric("Moderate metros",
              int(((sensitivity_df["rank_std"] > 15) & (sensitivity_df["rank_std"] <= 35)).sum()))
    k3.metric("Volatile metros (rank std > 35)",
              int((sensitivity_df["rank_std"] > 35).sum()))

    top_n = st.slider("Show top N metros", 10, 50, 25, key="sens_n")
    top_sens = sensitivity_df.head(top_n).copy()

    st.markdown("### 🌡️ Rank Heatmap — Top Metros Across Weight Scenarios")
    fig_sh = px.imshow(
        top_sens[scenario_value_cols],
        y=top_sens["metro_title"],
        color_continuous_scale=DESERT_DIVERGING_R,
        text_auto=True,
        aspect="auto",
        labels={"color": "Rank", "y": ""},
    )
    clean_layout(fig_sh, height=800)
    st.plotly_chart(fig_sh, use_container_width=True)

    st.markdown("### 🔀 Rank Volatility (Std Dev Across Scenarios)")
    sens_vol = sensitivity_df.sort_values("rank_std").copy()
    fig_vol = px.bar(
        sens_vol.head(30).iloc[::-1],
        x="rank_std", y="metro_title", orientation="h",
        color="stability",
        color_discrete_map={"Stable": SAGE_500, "Moderate": SAND_300, "Volatile": TERRA_500},
        labels={"rank_std": "Rank Std Dev (lower = more stable)", "metro_title": ""},
    )
    clean_layout(fig_vol, height=700)
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("### Weight Scenarios Compared")
    w_table = pd.DataFrame({
        "Scenario": ["Income-Heavy", "Employment-Heavy", "Balanced (default)", "Density-Heavy", "Diversity-Heavy"],
        "Income (%)": [40, 15, 25, 20, 20],
        "Unemployment (%)": [20, 40, 25, 15, 15],
        "Payroll/emp (%)": [15, 20, 20, 15, 15],
        "Bus. Density (%)": [15, 15, 15, 35, 15],
        "Diversity (%)": [10, 10, 15, 15, 35],
    })
    st.dataframe(w_table, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download sensitivity analysis (CSV)",
        data=sensitivity_df.to_csv(index=False).encode("utf-8"),
        file_name="sensitivity_analysis.csv", mime="text/csv",
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<p style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.8rem;">Model Performance & Explainability</p>', unsafe_allow_html=True)

    # ── helper: Plotly gauge ─────────────────────────────────────────────────
    def gauge_fig(value_str: str, title: str, color: str, fmt: str = ".1%") -> go.Figure:
        try:
            val = float(value_str)
        except (ValueError, TypeError):
            val = 0.0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"valueformat": fmt},
            title={"text": title, "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".0%"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0.0, 0.60], "color": "#fee2e2"},
                    {"range": [0.60, 0.80], "color": "#fef3c7"},
                    {"range": [0.80, 1.00], "color": "#d1fae5"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75, "value": 0.80,
                },
            },
        ))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📊 Classification — Predicting High-Opportunity Metros")

    best_clf  = extract_metric(classification_report_txt, "Best classifier")
    test_acc  = extract_metric(classification_report_txt, "Test Accuracy")
    test_f1   = extract_metric(classification_report_txt, "Test F1")
    test_auc  = extract_metric(classification_report_txt, "Test ROC-AUC")
    cv_f1_str = extract_metric(classification_report_txt, "5-fold CV F1")
    split_str = extract_metric(classification_report_txt, "Split")

    # — KPI gauges row --------------------------------------------------------
    ga, gb, gc, gd = st.columns([1.2, 1, 1, 1])
    with ga:
        st.markdown(f"**🏆 Best model**")
        st.markdown(f"### {best_clf}")
        if split_str:
            st.caption(f"Train / Val / Test: {split_str}")
        if cv_f1_str:
            st.info(f"5-Fold CV F1: **{cv_f1_str}**")
    with gb:
        st.plotly_chart(gauge_fig(test_acc, "Test Accuracy", SAGE_500), use_container_width=True)
    with gc:
        st.plotly_chart(gauge_fig(test_f1, "Test F1 Score", SAND_500), use_container_width=True)
    with gd:
        st.plotly_chart(gauge_fig(test_auc, "Test ROC-AUC", TERRA_600), use_container_width=True)

    # — All classifier candidates side by side --------------------------------
    st.markdown("### 🏁 All Classifier Candidates — Validation Comparison")
    clf_val_sorted_f1  = classifier_validation.sort_values("validation_f1")
    clf_val_sorted_auc = classifier_validation.sort_values("validation_auc")

    cc1, cc2 = st.columns(2)
    with cc1:
        fig_cv_f1 = px.bar(
            clf_val_sorted_f1, x="validation_f1", y="model", orientation="h",
            color="validation_f1", color_continuous_scale=DESERT_DIVERGING,
            range_color=[0.60, 1.0],
            text=[f"{v:.1%}" for v in clf_val_sorted_f1["validation_f1"]],
            labels={"validation_f1": "Validation F1", "model": ""},
            title="Validation F1 Score (higher = better)",
        )
        fig_cv_f1.update_traces(textposition="outside")
        clean_layout(fig_cv_f1, height=300)
        st.plotly_chart(fig_cv_f1, use_container_width=True)
    with cc2:
        fig_cv_auc = px.bar(
            clf_val_sorted_auc, x="validation_auc", y="model", orientation="h",
            color="validation_auc", color_continuous_scale=DESERT_SCALE,
            range_color=[0.90, 1.0],
            text=[f"{v:.1%}" for v in clf_val_sorted_auc["validation_auc"]],
            labels={"validation_auc": "Validation ROC-AUC", "model": ""},
            title="Validation ROC-AUC (higher = better)",
        )
        fig_cv_auc.update_traces(textposition="outside")
        clean_layout(fig_cv_auc, height=300)
        st.plotly_chart(fig_cv_auc, use_container_width=True)

    st.markdown("**Full candidate table**")
    st.dataframe(
        classifier_validation.rename(columns={
            "model": "Model", "validation_score": "Val Score",
            "validation_accuracy": "Val Accuracy", "validation_f1": "Val F1",
            "validation_auc": "Val AUC",
        }).style.format({
            "Val Score": "{:.4f}", "Val Accuracy": "{:.1%}",
            "Val F1": "{:.1%}", "Val AUC": "{:.1%}",
        }).highlight_max(subset=["Val F1", "Val AUC"], color="#d1fae5"),
        use_container_width=True, hide_index=True,
    )

    # — Per-class precision / recall / F1 -------------------------------------
    st.markdown("### 📋 Per-Class Precision / Recall / F1 (Test Set)")

    _report_rows: list[dict] = []
    for _line in classification_report_txt.split("\n"):
        _parts = _line.strip().split()
        if len(_parts) >= 4:
            try:
                _prec, _rec, _f1, _sup = float(_parts[-4]), float(_parts[-3]), float(_parts[-2]), int(_parts[-1])
                _label_raw = " ".join(_parts[:-4])
                if _label_raw in ("0", "1"):
                    _report_rows.append({
                        "Class": "Low/Medium" if _label_raw == "0" else "High",
                        "Precision": _prec, "Recall": _rec, "F1": _f1, "Support": _sup,
                    })
            except (ValueError, IndexError):
                pass

    if _report_rows:
        _report_df = pd.DataFrame(_report_rows)
        _melted = _report_df.melt(
            id_vars="Class", value_vars=["Precision", "Recall", "F1"],
            var_name="Metric", value_name="Score",
        )
        fig_per_class = px.bar(
            _melted, x="Metric", y="Score", color="Class", barmode="group",
            color_discrete_map={"Low/Medium": SAND_500, "High": SAGE_500},
            text=[f"{v:.1%}" for v in _melted["Score"]],
            range_y=[0, 1.15],
            labels={"Score": "Score", "Metric": ""},
            title="Classifier: Precision, Recall, F1 by Predicted Class",
        )
        fig_per_class.update_traces(textposition="outside")
        clean_layout(fig_per_class, height=420)
        st.plotly_chart(fig_per_class, use_container_width=True)

        st.dataframe(
            _report_df.style.format({
                "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}",
            }).highlight_max(subset=["Precision", "Recall", "F1"], color="#d1fae5"),
            use_container_width=True, hide_index=True,
        )

    # — Confusion matrix (raw + row-normalised) --------------------------------
    st.markdown("### 🧮 Confusion Matrix (Test Set)")
    _test_cases = outputs["classification_test_cases"]
    _cm_raw = pd.crosstab(_test_cases["actual_label"], _test_cases["predicted_label"])
    _cm_raw = _cm_raw.reindex(index=["Low/Medium", "High"], columns=["Low/Medium", "High"], fill_value=0)
    _cm_norm = _cm_raw.div(_cm_raw.sum(axis=1), axis=0).round(4)

    cm1, cm2 = st.columns(2)
    with cm1:
        fig_cm_raw = px.imshow(
            _cm_raw, text_auto=True, color_continuous_scale=DESERT_SCALE, aspect="auto",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title="Raw Counts",
        )
        clean_layout(fig_cm_raw, height=380)
        st.plotly_chart(fig_cm_raw, use_container_width=True)
    with cm2:
        fig_cm_norm = px.imshow(
            _cm_norm, text_auto=".1%", color_continuous_scale=DESERT_SCALE, aspect="auto",
            labels={"x": "Predicted", "y": "Actual", "color": "Rate"},
            title="Row-Normalised (Recall per Class)",
        )
        clean_layout(fig_cm_norm, height=380)
        st.plotly_chart(fig_cm_norm, use_container_width=True)

    # — Feature importances (Top 15) ------------------------------------------
    st.markdown("### 🎯 Feature Importance — Top 15")
    fi1, fi2 = st.columns(2)
    with fi1:
        st.markdown("**Classifier**")
        _ci = classifier_importance.head(15).sort_values("importance", key=lambda s: s.abs())
        fig_ci = px.bar(
            _ci, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=DESERT_DIVERGING,
            text=[f"{v:.3f}" for v in _ci["importance"]],
            labels={"importance": "Importance", "feature": ""},
        )
        fig_ci.update_traces(textposition="outside")
        clean_layout(fig_ci, height=580)
        st.plotly_chart(fig_ci, use_container_width=True)
    with fi2:
        st.markdown("**Regressor**")
        _ri = regressor_importance.head(15).sort_values("importance")
        fig_ri = px.bar(
            _ri, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=DESERT_SCALE,
            text=[f"{v:.3f}" for v in _ri["importance"]],
            labels={"importance": "Importance", "feature": ""},
        )
        fig_ri.update_traces(textposition="outside")
        clean_layout(fig_ri, height=580)
        st.plotly_chart(fig_ri, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — REGRESSION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 📈 Regression — Predicting Metro Unemployment Rate")

    best_reg   = extract_metric(regression_report_txt, "Best regressor")
    test_rmse  = extract_metric(regression_report_txt, "Test RMSE")
    test_mae   = extract_metric(regression_report_txt, "Test MAE")
    test_r2    = extract_metric(regression_report_txt, "Test R2")
    cv_r2_str  = extract_metric(regression_report_txt, "5-fold CV R2")

    rk1, rk2, rk3, rk4 = st.columns(4)
    rk1.metric("Best Regressor", best_reg)
    rk2.metric("Test RMSE", test_rmse, help="Root-mean-squared error (unemployment %, lower = better)")
    rk3.metric("Test MAE", test_mae,  help="Mean absolute error (unemployment %, lower = better)")
    rk4.metric("Test R²", test_r2,   help="Fraction of variance explained (higher = better)")
    if cv_r2_str:
        st.info(f"🔁 **5-Fold Cross-Validation R²:** {cv_r2_str}")

    rg1, rg2 = st.columns(2)
    with rg1:
        _rv_rmse = regressor_validation.sort_values("validation_rmse")
        fig_rv_rmse = px.bar(
            _rv_rmse, x="validation_rmse", y="model", orientation="h",
            color="validation_rmse", color_continuous_scale=DESERT_DIVERGING_R,
            text=[f"{v:.3f}" for v in _rv_rmse["validation_rmse"]],
            labels={"validation_rmse": "Validation RMSE (lower = better)", "model": ""},
            title="Validation RMSE (lower = better)",
        )
        fig_rv_rmse.update_traces(textposition="outside")
        clean_layout(fig_rv_rmse, height=280)
        st.plotly_chart(fig_rv_rmse, use_container_width=True)
    with rg2:
        _rv_r2 = regressor_validation.sort_values("validation_r2")
        fig_rv_r2 = px.bar(
            _rv_r2, x="validation_r2", y="model", orientation="h",
            color="validation_r2", color_continuous_scale=DESERT_SCALE,
            text=[f"{v:.3f}" for v in _rv_r2["validation_r2"]],
            labels={"validation_r2": "Validation R²", "model": ""},
            title="Validation R² (higher = better)",
        )
        fig_rv_r2.update_traces(textposition="outside")
        clean_layout(fig_rv_r2, height=280)
        st.plotly_chart(fig_rv_r2, use_container_width=True)

    st.markdown("**Full regressor candidate table**")
    st.dataframe(
        regressor_validation.rename(columns={
            "model": "Model", "validation_score": "Val Score",
            "validation_rmse": "Val RMSE", "validation_mae": "Val MAE", "validation_r2": "Val R²",
        }).style.format({
            "Val Score": "{:.4f}", "Val RMSE": "{:.4f}",
            "Val MAE": "{:.4f}", "Val R²": "{:.4f}",
        }).highlight_min(subset=["Val RMSE", "Val MAE"], color="#d1fae5")
          .highlight_max(subset=["Val R²"], color="#d1fae5"),
        use_container_width=True, hide_index=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C — MISCLASSIFIED CASES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## ❌ Misclassified Test Cases")
    if not misclassified_cases.empty:
        _fp = int((misclassified_cases["error_type"] == "False Positive").sum())
        _fn = int((misclassified_cases["error_type"] == "False Negative").sum())

        ep1, ep2, ep3 = st.columns(3)
        ep1.metric("Total Misclassified", len(misclassified_cases))
        ep2.metric("False Positives", _fp, help="Predicted High-Opportunity, actually Low/Medium")
        ep3.metric("False Negatives", _fn, help="Predicted Low/Medium, actually High-Opportunity")

        err_c1, err_c2 = st.columns([1, 2])
        with err_c1:
            fig_err_pie = px.pie(
                misclassified_cases, names="error_type",
                color="error_type",
                color_discrete_map={"False Positive": TERRA_500, "False Negative": SAND_500},
                title="Error Type Split",
                hole=0.4,
            )
            clean_layout(fig_err_pie, height=320)
            st.plotly_chart(fig_err_pie, use_container_width=True)
        with err_c2:
            _miss_disp = misclassified_cases[[
                "metro_title", "primary_state", "region",
                "actual_label", "predicted_label",
                "predicted_probability_high", "error_type", "opportunity_score",
            ]].copy()
            _miss_disp["predicted_probability_high"] = _miss_disp["predicted_probability_high"].map("{:.1%}".format)
            _miss_disp["opportunity_score"] = _miss_disp["opportunity_score"].map("{:.4f}".format)
            st.dataframe(_miss_disp, use_container_width=True, hide_index=True)

        # probability scatter for misclassified cases
        fig_err_scatter = px.scatter(
            misclassified_cases,
            x="opportunity_score", y="predicted_probability_high",
            color="error_type",
            color_discrete_map={"False Positive": TERRA_500, "False Negative": SAND_500},
            hover_name="metro_title",
            hover_data={"actual_label": True, "predicted_label": True, "region": True},
            labels={
                "opportunity_score": "Opportunity Score",
                "predicted_probability_high": "Predicted P(High)",
                "error_type": "Error Type",
            },
            title="Misclassified Cases: Opportunity Score vs Predicted Probability",
        )
        fig_err_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray",
                                   annotation_text="Decision boundary (0.5)")
        clean_layout(fig_err_scatter, height=420)
        st.plotly_chart(fig_err_scatter, use_container_width=True)
    else:
        st.success("No misclassified test cases — perfect classification on the held-out set.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION D — PER-FOLD (5-FOLD CV) MISCLASSIFIED CASES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🔁 Per-Fold Misclassified Cases (5-Fold Cross-Validation)")
    st.caption("These errors occurred during 5-fold CV on the training+validation set (60%+20%). The final test set errors are shown above.")

    if not cv_fold_misclassified.empty:
        fold_summary = (
            cv_fold_misclassified.groupby("fold")
            .agg(
                total_errors=("metro_title", "count"),
                false_positives=("error_type", lambda x: (x == "False Positive").sum()),
                false_negatives=("error_type", lambda x: (x == "False Negative").sum()),
            )
            .reset_index()
        )

        fold_cols = st.columns(5)
        for _, frow in fold_summary.iterrows():
            fold_cols[int(frow["fold"]) - 1].metric(
                f"Fold {int(frow['fold'])}",
                f"{int(frow['total_errors'])} errors",
                help=f"FP: {int(frow['false_positives'])}  FN: {int(frow['false_negatives'])}",
            )

        st.markdown(f"**Total misclassifications across all 5 folds: {len(cv_fold_misclassified)}**")

        selected_fold = st.selectbox(
            "Filter by fold", ["All folds"] + [f"Fold {i}" for i in range(1, 6)]
        )
        fold_disp = cv_fold_misclassified.copy()
        if selected_fold != "All folds":
            fold_num_sel = int(selected_fold.split()[1])
            fold_disp = fold_disp[fold_disp["fold"] == fold_num_sel]

        fold_disp_show = fold_disp[[
            "fold", "metro_title", "primary_state", "region",
            "actual_label", "predicted_label",
            "predicted_probability_high", "error_type", "opportunity_score",
        ]].copy()
        fold_disp_show["predicted_probability_high"] = fold_disp_show["predicted_probability_high"].map("{:.1%}".format)
        fold_disp_show["opportunity_score"] = fold_disp_show["opportunity_score"].map("{:.4f}".format)
        st.dataframe(fold_disp_show, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download CV fold misclassifications",
            fold_disp.to_csv(index=False),
            "cv_fold_misclassified.csv",
            "text/csv",
        )
    else:
        st.info("Run `python3 main.py` to generate per-fold misclassification data.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION E — GEOGRAPHIC VIEW OF FOLD SPLITS & MISCLASSIFICATIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🗺️ Geographic View — Splits & Misclassifications")
    st.caption("Pick a fold (or the final test set) to see how metros were split and which ones the model got wrong.")

    if not cv_fold_assignments.empty and "lat" in cv_fold_assignments.columns:
        scenario_choice = st.selectbox(
            "Scenario",
            ["Test Set", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
            key="fold_map_scenario",
        )

        scn = cv_fold_assignments[cv_fold_assignments["scenario"] == scenario_choice].copy()
        scn = scn.dropna(subset=["lat", "lon"])

        # Build a single category column combining split + misclassification flag
        def _category(row):
            if row["split"] == "Training":
                return "Training"
            label = "Test" if row["split"] == "Test" else "Validation"
            if row["is_misclassified"]:
                return f"{label} — ❌ Misclassified"
            return f"{label} — ✅ Correct"

        scn["category"] = scn.apply(_category, axis=1)
        scn["marker_size"] = np.where(scn["is_misclassified"], 14, 6)

        color_map = {
            "Training": "#4F8EF7",
            "Validation — ✅ Correct": "#F2A65A",
            "Validation — ❌ Misclassified": "#D7263D",
            "Test — ✅ Correct": "#F2A65A",
            "Test — ❌ Misclassified": "#D7263D",
        }

        # Split summary metrics
        cnt_train = int((scn["split"] == "Training").sum())
        cnt_eval = int((scn["split"] != "Training").sum())
        cnt_misc = int(scn["is_misclassified"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Training metros", cnt_train)
        m2.metric(
            "Validation metros" if scenario_choice != "Test Set" else "Test metros",
            cnt_eval,
        )
        m3.metric("Misclassified", cnt_misc)

        fig_map = px.scatter_geo(
            scn,
            lat="lat", lon="lon",
            color="category",
            size="marker_size",
            size_max=18,
            color_discrete_map=color_map,
            hover_name="metro_title",
            hover_data={
                "primary_state": True,
                "split": True,
                "actual_label": True,
                "predicted_label": True,
                "predicted_probability_high": ":.1%",
                "error_type": True,
                "lat": False, "lon": False, "marker_size": False, "category": False,
            },
            scope="usa",
            title=f"{scenario_choice} — Train/Validation/Test split + misclassifications",
        )
        fig_map.update_traces(marker=dict(line=dict(width=0)))
        # Add a black ring around misclassified markers
        for tr in fig_map.data:
            if "Misclassified" in tr.name:
                tr.marker.line = dict(width=2, color="black")
        clean_layout(fig_map, height=560)
        st.plotly_chart(fig_map, use_container_width=True)

        st.caption(
            "🔵 Blue = Training metros · 🟠 Orange = Validation/Test metros correctly classified · "
            "🔴 Red (larger, black ring) = Misclassified"
        )
    else:
        st.info("Geographic data not available — re-run `python3 main.py` to regenerate.")
