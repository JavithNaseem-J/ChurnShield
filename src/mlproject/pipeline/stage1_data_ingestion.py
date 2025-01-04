from mlproject.config.config import ConfigurationManager
from mlproject.components.data_ingestion import DataIngestion
from mlproject import logger

STAGE = "Data Ingestion Stage"


class DataIngestiontrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
    
if __name__ == "__main__":
    try:
        logger.info(f"Running {STAGE}...")
        pipeline = DataIngestiontrainingPipeline()
        pipeline.main()
        logger.info(f"{STAGE} completed successfully!")

    except Exception as e:
        logger.error(f"{STAGE} failed! Error: {e}")
        raise e
   

