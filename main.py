from mlproject import logger
from mlproject.pipeline.stage1_data_ingestion import DataIngestiontrainingPipeline
from mlproject.pipeline.stage2_data_validation import DataValidationtrainingPipeline


STAGE = "Data Ingestion Stage"

try:
    logger.info(f"Running {STAGE}...")
    pipeline = DataIngestiontrainingPipeline()
    pipeline.main()
    logger.info(f"{STAGE} completed successfully!")

except Exception as e:
    logger.error(f"{STAGE} failed! Error: {e}")
    raise e

STAGE = "Data Validation Stage"

try:
    logger.info(f"Running {STAGE}...")
    pipeline = DataValidationtrainingPipeline()
    pipeline.main()
    logger.info(f"{STAGE} completed successfully!")

except Exception as e:
    logger.error(f"{STAGE} failed! Error: {e}")
    raise e