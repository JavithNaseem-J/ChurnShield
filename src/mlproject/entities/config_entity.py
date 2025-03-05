from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    target_column: str
    label_encoder: Path
    feature_encoder: Path  
    preprocessor_path: Path
    columns_to_drop: list
    num_cols: list
    cat_cols: list



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    subsample: float
    num_leaves: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    lambda_l2: float
    lambda_l1: float
    colsample_bytree: float



@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    metric_file_path: Path
    preprocessor_path: Path
    test_raw_data: Path  
    target_column: str
    all_params: dict

