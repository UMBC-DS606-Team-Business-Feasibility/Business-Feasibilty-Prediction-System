import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names, out_path: str):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    order = np.argsort(importances)

    plt.figure(figsize=(9, 5))
    plt.barh([feature_names[i] for i in order], importances[order])
    plt.title("Feature Importance (Best Classifier)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def shap_summary(model, X_test, out_path: str):
    try:
        import shap
    except ImportError:
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception:
        return
