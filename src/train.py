import os
import logging
import pandas as pd
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def spark():
    return (
        SparkSession.builder
        .appName("Retail Forecast - Train")
        .getOrCreate()
    )


def train_models(gold_path: str, label_col: str = "daily_qty"):

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    s = spark()

    logger.info("Reading Gold dataset...")
    df = s.read.parquet(gold_path).dropna()

    feature_cols = [
        c for c in df.columns
        if c.startswith("lag_") or c.startswith("roll_")
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    data = assembler.transform(df).select(
        "features",
        col(label_col).alias("label")
    )

    train, test = data.randomSplit([0.8, 0.2], seed=42)

    evaluator_rmse = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )

    evaluator_mae = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="mae"
    )

    results = {}

    # ==========================
    # Linear Regression
    # ==========================
    with mlflow.start_run(run_name="Linear_Regression"):

        lr = LinearRegression(
            featuresCol="features",
            labelCol="label"
        )

        lr_model = lr.fit(train)
        lr_pred = lr_model.transform(test)

        rmse_lr = float(evaluator_rmse.evaluate(lr_pred))
        mae_lr = float(evaluator_mae.evaluate(lr_pred))

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse_lr)
        mlflow.log_metric("mae", mae_lr)

        mlflow.spark.log_model(lr_model, "model")

        lr_pred.select("label", "prediction") \
            .write.mode("overwrite") \
            .parquet("reports/lr_predictions")

        lr_model.write().overwrite().save("models/linear_regression")

        results["Linear_Regression"] = {
            "rmse": rmse_lr,
            "mae": mae_lr
        }

    # ==========================
    # Random Forest
    # ==========================
    with mlflow.start_run(run_name="Random_Forest"):

        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol="label",
            numTrees=200,
            maxDepth=10,
            seed=42
        )

        rf_model = rf.fit(train)
        rf_pred = rf_model.transform(test)

        rmse_rf = float(evaluator_rmse.evaluate(rf_pred))
        mae_rf = float(evaluator_mae.evaluate(rf_pred))

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("numTrees", 200)
        mlflow.log_param("maxDepth", 10)

        mlflow.log_metric("rmse", rmse_rf)
        mlflow.log_metric("mae", mae_rf)

        mlflow.spark.log_model(rf_model, "model")

        rf_pred.select("label", "prediction") \
            .write.mode("overwrite") \
            .parquet("reports/rf_predictions")

        rf_model.write().overwrite().save("models/random_forest")

        # Feature Importance Save
        importances = rf_model.featureImportances.toArray()
        importance_df = pd.DataFrame({
            "feature_name": feature_cols,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        importance_df.to_csv("reports/feature_importance.csv", index=False)

        results["Random_Forest"] = {
            "rmse": rmse_rf,
            "mae": mae_rf
        }

    # Save Model Comparison Metrics
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv("reports/model_metrics.csv")

    logger.info(f"Training complete. Results: {results}")

    return results


if __name__ == "__main__":
    gold_input = "data/gold/features_sql"
    train_models(gold_input)
