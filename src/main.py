import logging
from .ingest import ingest_raw_data
from .transform import transform_data
from .features import build_gold_features


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Paths
    raw_file = "data/raw/Online_Retail.xlsx"
    bronze_path = "data/bronze/online_retail"
    silver_path = "data/silver/online_retail_cleaned"
    gold_df_path = "data/gold/features_dataframe"
    gold_sql_path = "data/gold/features_sql"

    logger.info("STEP 1: Ingestion → Bronze")
    ingest_raw_data(raw_file, bronze_path)

    logger.info("STEP 2: Transformation → Silver")
    transform_data(bronze_path, silver_path)

    logger.info("STEP 3: Feature Engineering → Gold (DF + SQL)")
    build_gold_features(silver_path, gold_df_path, gold_sql_path)

    logger.info("PIPELINE COMPLETED ✅")
