# Metropolitan Business Feasibility Prediction System
## Final Submission Report

---

## Project Status
- **Status:** complete and submission-ready
- **Final metro sample:** all 386 overlapping metros
- **Raw CBP rows used:** 369,496
- **Business categories supported:** 9
- **Total model features:** 34

---

## Data Stack (4 official sources)

| Dataset | Year | Provider |
|---------|------|----------|
| County Business Patterns (CBP) | **2023** | U.S. Census Bureau |
| American Community Survey (ACS) | 2024 | U.S. Census Bureau |
| BLS Metro Unemployment | 2020-2024 | Bureau of Labor Statistics |
| **BEA Regional Price Parities** | **2024** | Bureau of Economic Analysis |

---

## Modeling Summary

### Classification (high-opportunity metros)
- **Best classifier:** LogisticRegression (tuned)
- **Test Accuracy:** 0.8333
- **Test F1:** 0.7451
- **Test ROC-AUC:** 0.9020

### Regression (unemployment rate)
- **Best regressor:** RandomForestRegressor
- **Test RMSE:** 1.4007
- **Test MAE:** 0.7643
- **Test R2:** 0.5128

---

## Opportunity Score (Phase 5 formula)

```
0.25 * real_median_income_norm     (BEA-RPP-adjusted)
0.20 * inverse_unemployment_norm
0.15 * annual_payroll_per_employee_norm
0.15 * establishments_per_10k_norm
0.15 * industry_entropy_norm
0.10 * affordability_norm           (cost-of-living penalty)
```

Cost-of-living adjustment shifted top rankings: SF #2->#4, Boston top-5->#13,
Midland TX -> #2, Sioux Falls and Bismarck entered top 6.

---

## Phase Delivery Summary
- **Phase 1**: 32-feature pipeline, payroll/labour ratios, 6 categories
- **Phase 2**: 4-source merge, opportunity score, classifier+regressor
- **Phase 3**: entry-strategy labels, white-space scoring, city aliases, what-if lab
- **Phase 4**: GridSearchCV tuning, VotingClassifier, calibrated probs, K-Means
  clustering, PCA, sensitivity analysis, 9 categories, 8-tab BI dashboard
- **Phase 5**: BEA RPP cost-of-living + housing, CBP 2022->2023 refresh, lat/lon
  bubble map, executive PowerPoint generator, dashboard cost-of-living surfacing

---

## Key Files
- `README.md` - project overview and usage
- `CHANGELOG.md` - full history across 5 phases
- `capstone_pipeline.py` - core pipeline (~2,200 lines)
- `app.py` - Streamlit BI dashboard, 8 interactive tabs
- `exec_report.py` - executive PowerPoint and PDF generator
- `main.py` - entry point
- `outputs/` - 50+ generated CSVs, PNGs, and reports
- `outputs/executive_summary.pptx` - submission-ready slide deck
- `outputs/executive_summary.pdf` - one-page PDF summary

---

## Limitations
- CBP is 2023; ACS, BLS, and BEA RPP are 2024 (intentional given staggered release)
- Metro-level feasibility, not a business-level causal model
- Opportunity score is decision-support; sensitivity analysis exposes weight choices

---

## How to Reproduce
```bash
pip install -r requirements.txt
python3 main.py
streamlit run app.py
```