import os


class Paths:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    YELP_JSON = os.path.join(DATA_DIR, "yelp_business.json")
    CENSUS_CSV = os.path.join(DATA_DIR, "census_income.csv")
    BLS_XLSX = os.path.join(DATA_DIR, "bls_unemployment.xlsx")

    MASTER_CSV = os.path.join(RESULTS_DIR, "master_city_dataset.csv")
    RANKING_CSV = os.path.join(RESULTS_DIR, "city_ranking.csv")

    CLASSIFICATION_REPORT_TXT = os.path.join(
        RESULTS_DIR, "classification_report.txt")
    REGRESSION_REPORT_TXT = os.path.join(RESULTS_DIR, "regression_report.txt")
    SUMMARY_TXT = os.path.join(RESULTS_DIR, "summary.txt")

    FEATURE_IMPORTANCE_PNG = os.path.join(
        RESULTS_DIR, "feature_importance.png")
    SHAP_SUMMARY_PNG = os.path.join(RESULTS_DIR, "shap_summary.png")

    MODEL_BUNDLE = os.path.join(MODELS_DIR, "model_bundle.joblib")

    @staticmethod
    def ensure_dirs():
        os.makedirs(Paths.DATA_DIR, exist_ok=True)
        os.makedirs(Paths.RESULTS_DIR, exist_ok=True)
        os.makedirs(Paths.MODELS_DIR, exist_ok=True)
