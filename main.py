from mlproject import logger
from mlproject.pipeline.stage1_data_ingestion import DataIngestiontrainingPipeline
from mlproject.pipeline.stage2_data_validation import DataValidationtrainingPipeline
from mlproject.pipeline.stage3_data_transformation import DataTransformationTrainingPipeline
from mlproject.pipeline.stage4_modeltraining import ModelTrainingPipeline


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


STAGE = "Data Transformation stage"

try:
    logger.info(f"Running {STAGE}...")
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()
    logger.info(f"{STAGE} completed successfully!")

except Exception as e:
    logger.error(f"{STAGE} failed! Error: {e}")
    raise e

STAGE = "Model Training Stage"

try:
    logger.info(f"Running {STAGE}...")
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logger.info(f"{STAGE} completed successfully!")

except Exception as e:
    logger.error(f"{STAGE} failed! Error: {e}")
    raise e
