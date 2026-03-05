import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

FEATURES = ["avg_rating", "avg_review_count",
            "business_count", "median_income", "employment_rate"]


def _adaptive_cv(n_train: int, max_cv: int = 5) -> int:
    return max(2, min(max_cv, n_train)) if n_train >= 2 else 1


def run_classification_models(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["class_label"]

    test_size = 0.30 if len(df) >= 10 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    cv = _adaptive_cv(len(X_train), max_cv=5)

    best_name, best_model, best_score = None, None, -1.0
    cv_scores = {}

    for name, model in models.items():
        if cv >= 2 and len(X_train) >= cv:
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="accuracy")
            cv_scores[name] = float(scores.mean())
        else:
            cv_scores[name] = 0.0

        if cv_scores[name] > best_score:
            best_score = cv_scores[name]
            best_name = name
            best_model = model

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    auc = None
    try:
        if hasattr(best_model, "predict_proba") and len(set(y_test)) > 1:
            auc = float(roc_auc_score(
                y_test, best_model.predict_proba(X_test)[:, 1]))
    except Exception:
        auc = None

    return {
        "best_name": best_name,
        "best_model": best_model,
        "cv_scores": cv_scores,
        "report": classification_report(y_test, y_pred),
        "roc_auc": auc,
        "X_test": X_test,
        "y_test": y_test,
        "feature_columns": FEATURES,
    }


def run_regression_models(df: pd.DataFrame):
    data = df.copy()
    # Revenue proxy (realistic for capstone when revenue isn't available)
    data["revenue_proxy"] = data["business_count"] * \
        data["avg_review_count"] * data["avg_rating"]

    X = data[FEATURES]
    y = data["revenue_proxy"]

    test_size = 0.30 if len(data) >= 10 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
    }

    cv = _adaptive_cv(len(X_train), max_cv=5)

    best_name, best_model, best_score = None, None, -999.0
    cv_scores = {}

    for name, model in models.items():
        if cv >= 2 and len(X_train) >= cv:
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="r2")
            cv_scores[name] = float(scores.mean())
        else:
            cv_scores[name] = -999.0

        if cv_scores[name] > best_score:
            best_score = cv_scores[name]
            best_name = name
            best_model = model

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    report = (
        f"Best Regression Model: {best_name}\n"
        f"CV R2 (cv={cv}): {best_score:.4f}\n"
        f"Test RMSE: {rmse:.4f}\n"
        f"Test MAE: {mae:.4f}\n"
        f"Test R2: {r2:.4f}\n"
    )

    return {
        "best_name": best_name,
        "best_model": best_model,
        "cv_scores": cv_scores,
        "report": report,
        "target_columns": ["revenue_proxy"],
    }


def evaluate_ranking_spearman(df: pd.DataFrame):
    tmp = df.copy()
    tmp["revenue_proxy"] = tmp["business_count"] * \
        tmp["avg_review_count"] * tmp["avg_rating"]
    corr, p = spearmanr(tmp["feasibility_score"], tmp["revenue_proxy"])
    return {"spearman_corr": float(corr), "p_value": float(p)}


def save_model_bundle(best_clf, best_reg, out_path: str, feature_columns, target_columns):
    bundle = {
        "classifier": best_clf,
        "regressor": best_reg,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
    }
    joblib.dump(bundle, out_path)
