import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


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
        .appName("Retail Demand Forecasting - Transformation") \
        .getOrCreate()


# -----------------------------
# Transformation Function
# -----------------------------
def transform_data(input_path: str, output_path: str):

    spark = create_spark_session()

    logger.info("Reading Bronze layer...")
    df = spark.read.parquet(input_path)

    initial_count = df.count()
    logger.info(f"Initial record count: {initial_count}")

    # Remove rows with null customer_id
    df = df.filter(col("customerid").isNotNull())

    # Remove negative or zero quantities
    df = df.filter(col("quantity") > 0)

    # Remove cancellations (InvoiceNo starting with 'C')
    df = df.filter(~col("invoiceno").startswith("C"))

    # Add revenue column
    df = df.withColumn("revenue", col("quantity") * col("unitprice"))

    final_count = df.count()
    logger.info(f"Final record count after cleaning: {final_count}")

    logger.info("Writing Silver layer...")
    df.write.mode("overwrite").parquet(output_path)

    logger.info("Transformation completed successfully!")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":

    bronze_input = "data/bronze/online_retail"
    silver_output = "data/silver/online_retail_cleaned"

    transform_data(bronze_input, silver_output)
