from mlproject.config.config import ConfigurationManager
from mlproject.components.data_modelevaluation import ModelEvaluation
from mlproject import logger


STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.save_results()

if __name__ == '__main__':
    try:
        logger.info(f"Running {STAGE_NAME}...")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(f"{STAGE_NAME} completed successfully!")
    except Exception as e:
        logger.error(f"{STAGE_NAME} failed! Error: {e}")