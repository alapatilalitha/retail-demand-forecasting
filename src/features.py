import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, sum as _sum,
    lag, avg
)
from pyspark.sql.window import Window


# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Spark Session
# -----------------------------
def create_spark_session():
    return SparkSession.builder \
        .appName("Retail Demand Forecasting - Gold Features") \
        .getOrCreate()


# -----------------------------
# Gold Feature Engineering
# -----------------------------
def build_gold_features(silver_path: str, gold_df_path: str, gold_sql_path: str):

    spark = create_spark_session()

    logger.info("Reading Silver layer...")
    df = spark.read.parquet(silver_path)

    # Ensure invoice_date is date type
    # Your column name likely "invoicedate" after standardization
    df = df.withColumn("invoice_date", to_date(col("invoicedate")))

    logger.info("Building daily aggregation (daily revenue & daily quantity)...")
    daily = (
        df.groupBy("invoice_date")
          .agg(
              _sum(col("quantity")).alias("daily_qty"),
              _sum(col("revenue")).alias("daily_revenue")
          )
          .orderBy("invoice_date")
    )

    # -----------------------------
    # 1) DataFrame API Features
    # -----------------------------
    logger.info("Creating features using DataFrame API (lags + rolling)...")

    w = Window.orderBy("invoice_date")

    # rolling window needs ROWS BETWEEN
    w_7 = w.rowsBetween(-6, 0)    # last 7 days incl today
    w_14 = w.rowsBetween(-13, 0)  # last 14 days incl today

    daily_df_features = (
        daily
        .withColumn("lag_qty_1", lag(col("daily_qty"), 1).over(w))
        .withColumn("lag_qty_7", lag(col("daily_qty"), 7).over(w))
        .withColumn("roll_qty_7", avg(col("daily_qty")).over(w_7))
        .withColumn("roll_qty_14", avg(col("daily_qty")).over(w_14))
        .withColumn("lag_rev_1", lag(col("daily_revenue"), 1).over(w))
        .withColumn("lag_rev_7", lag(col("daily_revenue"), 7).over(w))
        .withColumn("roll_rev_7", avg(col("daily_revenue")).over(w_7))
        .withColumn("roll_rev_14", avg(col("daily_revenue")).over(w_14))
    )

    logger.info("Writing Gold (DataFrame API) features...")
    daily_df_features.write.mode("overwrite").parquet(gold_df_path)

    # -----------------------------
    # 2) Spark SQL Features
    # -----------------------------
    logger.info("Creating features using Spark SQL (window functions)...")

    daily.createOrReplaceTempView("daily_sales")

    sql_features = spark.sql("""
        WITH base AS (
            SELECT
                invoice_date,
                daily_qty,
                daily_revenue
            FROM daily_sales
        )
        SELECT
            invoice_date,
            daily_qty,
            daily_revenue,

            LAG(daily_qty, 1) OVER (ORDER BY invoice_date) AS lag_qty_1,
            LAG(daily_qty, 7) OVER (ORDER BY invoice_date) AS lag_qty_7,
            AVG(daily_qty) OVER (ORDER BY invoice_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS roll_qty_7,
            AVG(daily_qty) OVER (ORDER BY invoice_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS roll_qty_14,

            LAG(daily_revenue, 1) OVER (ORDER BY invoice_date) AS lag_rev_1,
            LAG(daily_revenue, 7) OVER (ORDER BY invoice_date) AS lag_rev_7,
            AVG(daily_revenue) OVER (ORDER BY invoice_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS roll_rev_7,
            AVG(daily_revenue) OVER (ORDER BY invoice_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS roll_rev_14

        FROM base
        ORDER BY invoice_date
    """)

    logger.info("Writing Gold (Spark SQL) features...")
    sql_features.write.mode("overwrite").parquet(gold_sql_path)

    logger.info("Gold feature engineering completed successfully!")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":

    silver_input = "data/silver/online_retail_cleaned"

    gold_df_output = "data/gold/features_dataframe"
    gold_sql_output = "data/gold/features_sql"

    build_gold_features(silver_input, gold_df_output, gold_sql_output)
