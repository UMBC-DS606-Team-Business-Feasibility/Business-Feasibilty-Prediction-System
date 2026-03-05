from src.report import save_outputs
from src.explainability import plot_feature_importance, shap_summary
from src.modeling import (
    run_classification_models,
    run_regression_models,
    evaluate_ranking_spearman,
    save_model_bundle,
)
from src.scoring import add_feasibility_score, add_class_label
from src.merge import build_master_city_table
from src.bls_cleaner import bls_laus_metro_to_city_employment
from src.census_cleaner import census_b19013_to_city_income
from src.yelp_features import clean_yelp_business, city_level_yelp_features
from src.data_loaders import load_yelp_business_ndjson, load_census_csv, load_bls_xlsx
from src.config import Paths
from src.city_lookup import lookup_city_score
from src.city_compare import compare_cities
from src.eda_analysis import run_eda
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    Paths.ensure_dirs()

    print("Loading raw data...")
    yelp_raw = load_yelp_business_ndjson(Paths.YELP_JSON)
    census_raw = load_census_csv(Paths.CENSUS_CSV)
    bls_raw = load_bls_xlsx(Paths.BLS_XLSX)

    print("Cleaning + feature engineering...")
    yelp = clean_yelp_business(yelp_raw)
    yelp_city = city_level_yelp_features(yelp)

    census_city = census_b19013_to_city_income(census_raw)
    bls_city = bls_laus_metro_to_city_employment(bls_raw, target_year=2024)

    print("Merging datasets...")
    master = build_master_city_table(
        yelp_city=yelp_city, census_city=census_city, bls_city=bls_city)
    print(f"Master rows: {len(master)} | Cities: {master['city'].nunique()}")

    print("Feasibility score + labels...")
    master = add_feasibility_score(master)
    master = add_class_label(master, threshold=0.60)
    # Save dataset for dashboard
    master.to_csv("outputs/master_dataset.csv", index=False)

    print("Modeling (classification + regression)...")
    clf_results = run_classification_models(master)
    reg_results = run_regression_models(master)
    rank_stats = evaluate_ranking_spearman(master)

    print("Saving best model bundle...")
    save_model_bundle(
        clf_results["best_model"],
        reg_results["best_model"],
        Paths.MODEL_BUNDLE,
        feature_columns=clf_results["feature_columns"],
        target_columns=reg_results["target_columns"],
    )

    print("Explainability outputs...")
    plot_feature_importance(
        clf_results["best_model"],
        clf_results["feature_columns"],
        Paths.FEATURE_IMPORTANCE_PNG
    )
    shap_summary(
        clf_results["best_model"],
        clf_results["X_test"],
        Paths.SHAP_SUMMARY_PNG
    )

    print("Saving results + reports...")
    save_outputs(master=master, clf_results=clf_results,
                 reg_results=reg_results, rank_stats=rank_stats)

# Run EDA
    run_eda(master)

# Interactive lookup
    lookup_city_score(master)

# Compare cities
    compare_cities(master)


if __name__ == "__main__":
    main()
