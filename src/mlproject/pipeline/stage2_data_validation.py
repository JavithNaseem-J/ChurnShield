from mlproject.config.config import ConfigurationManager
from mlproject.components.data_validation import DataValiadtion
from mlproject import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationtrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()


if __name__ == "__main__":
    try:
        logger.info(f"Running {STAGE_NAME}...")
        pipeline = DataValidationtrainingPipeline()
        pipeline.main()
        logger.info(f"{STAGE_NAME} completed successfully!")
    except Exception as e:
        logger.error(f"{STAGE_NAME} failed! Error: {e}")
        raise e