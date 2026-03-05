import pandas as pd
from src.config import Paths


def save_outputs(master: pd.DataFrame, clf_results: dict, reg_results: dict, rank_stats: dict):
    master.to_csv(Paths.MASTER_CSV, index=False)

    ranking = master[["city", "feasibility_score"]].sort_values(
        "feasibility_score", ascending=False)
    ranking.to_csv(Paths.RANKING_CSV, index=False)

    with open(Paths.CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"Best Classifier: {clf_results['best_name']}\n")
        f.write("CV Accuracy:\n")
        for k, v in clf_results["cv_scores"].items():
            f.write(f"  {k}: {v:.4f}\n")
        if clf_results["roc_auc"] is not None:
            f.write(f"ROC-AUC: {clf_results['roc_auc']:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(clf_results["report"])

    with open(Paths.REGRESSION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(reg_results["report"])

    with open(Paths.SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("Business Feasibility Prediction System - Summary\n")
        f.write("------------------------------------------------\n")
        f.write(f"Rows in master dataset: {len(master)}\n")
        f.write(f"Unique cities: {master['city'].nunique()}\n")
        f.write(
            f"Spearman correlation (feasibility_score vs revenue_proxy): {rank_stats['spearman_corr']:.4f}\n")
        f.write(f"P-value: {rank_stats['p_value']:.6f}\n")
        f.write("\nOutputs created in results/.\n")
