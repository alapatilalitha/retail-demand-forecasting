import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_reports_folder():
    if not os.path.exists("reports"):
        os.makedirs("reports")


def plot_model_comparison():
    path = "reports/model_metrics.csv"

    if not os.path.exists(path):
        logger.warning("model_metrics.csv not found.")
        return

    df = pd.read_csv(path)

    plt.figure(figsize=(8,5))
    plt.bar(df["Unnamed: 0"], df["rmse"])
    plt.title("Model Comparison (RMSE)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig("reports/model_comparison.png")
    plt.close()

    logger.info("Model comparison chart saved.")


def plot_feature_importance():
    path = "reports/feature_importance.csv"

    if not os.path.exists(path):
        logger.warning("Feature importance file not found.")
        return

    df = pd.read_csv(path).head(10)

    plt.figure(figsize=(8,6))
    plt.barh(df["feature_name"], df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png")
    plt.close()

    logger.info("Feature importance chart saved.")


def plot_rf_predictions():

    spark = SparkSession.builder.appName("Visualization").getOrCreate()
    df = spark.read.parquet("reports/rf_predictions")

    pdf = df.limit(100).toPandas()

    # Scatter
    plt.figure(figsize=(6,6))
    plt.scatter(pdf["label"], pdf["prediction"], alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Scatter")
    plt.tight_layout()
    plt.savefig("reports/scatter_plot.png")
    plt.close()

    # Residuals
    residuals = pdf["label"] - pdf["prediction"]

    plt.figure(figsize=(6,4))
    plt.scatter(pdf["prediction"], residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig("reports/residual_plot.png")
    plt.close()

    logger.info("Prediction visualizations saved.")


if __name__ == "__main__":

    ensure_reports_folder()

    plot_model_comparison()
    plot_feature_importance()
    plot_rf_predictions()

    logger.info("All visualizations generated.")
