
import logging
import pandas as pd
from pyspark.sql import SparkSession

print("INGEST SCRIPT STARTED")

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Create Spark Session
# -----------------------------
def create_spark_session():
    return SparkSession.builder \
        .appName("Retail Demand Forecasting - Ingestion") \
        .getOrCreate()


# -----------------------------
# Ingestion Function
# -----------------------------
def ingest_raw_data(input_path: str, output_path: str):

    logger.info("Reading raw Excel file using Pandas...")

    # Read Excel file
    pdf = pd.read_excel(input_path)

    logger.info("Converting to Spark DataFrame...")
    spark = create_spark_session()
    df = spark.createDataFrame(pdf)

    logger.info("Standardizing column names...")

    for column in df.columns:
        df = df.withColumnRenamed(column, column.strip().lower().replace(" ", "_"))

    logger.info("Writing Bronze layer as Parquet...")

    df.write.mode("overwrite").parquet(output_path)

    logger.info("Ingestion completed successfully!")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":

    input_file = "data/raw/Online_Retail.xlsx"
    bronze_output = "data/bronze/online_retail"

    ingest_raw_data(input_file, bronze_output)
