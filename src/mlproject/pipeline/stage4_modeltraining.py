from mlproject.config.config import ConfigurationManager
from mlproject.components.data_modeltraining import ModelTrainer
from mlproject import logger



STAGE_NAME = "Model Trainer stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()


if __name__ == '__main__':
    try:
        logger.info(f"Running {STAGE_NAME}...")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f"{STAGE_NAME} completed successfully!")
    except Exception as e:
        logger.error(f"{STAGE_NAME} failed! Error: {e}")