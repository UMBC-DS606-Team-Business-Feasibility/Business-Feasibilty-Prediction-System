import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df):

    top = df.sort_values("feasibility_score", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="feasibility_score", y="city")
    plt.title("Top Cities for Business Feasibility")
    plt.tight_layout()
    plt.savefig("outputs/top_cities.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="median_income",
        y="feasibility_score"
    )
    plt.title("Income vs Feasibility")
    plt.savefig("outputs/income_vs_feasibility.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    corr = df[
        [
            "avg_rating",
            "avg_review_count",
            "business_count",
            "median_income",
            "employment_rate",
            "feasibility_score"
        ]
    ].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()
