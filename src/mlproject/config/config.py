from src.mlproject.constants import *
from src.mlproject.utils.common import read_yaml, create_directories
from src.mlproject.entities.config_entity import (DataIngestionConfig, 
                                                DataValidationConfig, 
                                                DataTransformationConfig,
                                                ModelTrainerConfig,
                                                ModelEvaluationConfig,
                                                ModelMonitoringConfig)
import os


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        schema = self.schema
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            target_column=config.target_column,
            preprocessor_path=config.preprocessor_path,
            label_encoder=config.label_encoder,
            columns_to_drop=schema.columns_to_drop,
            num_cols=schema.num_cols,
            cat_cols=schema.cat_cols
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.LGBMClassifier

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            subsample=params.subsample,
            num_leaves=params.num_leaves,
            learning_rate=params.learning_rate,
            lambda_l2=params.lambda_l2,
            lambda_l1=params.lambda_l1,
            colsample_bytree=params.colsample_bytree
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.LGBMClassifier
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_raw_data=config.test_raw_data,
            model_path=config.model_path,
            all_params=params,
            metric_file_path=config.metric_file_path,
            preprocessor_path=config.preprocessor_path,
            target_column=schema.name
        )

        return model_evaluation_config
    
    def get_model_monitoring_config(self) -> ModelMonitoringConfig:
        config = self.config.model_monitoring
        
        create_directories([config.root_dir])
        
        model_monitoring_config = ModelMonitoringConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            preprocessor_path=config.preprocessor_path
        )
        
        return model_monitoring_config