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
    STATUS_FILE: Path
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    target_column: str
    label_encoder: Path
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
    n_estimators: int
    max_depth: int
    subsample: float
    num_leaves: int
    learning_rate: float
    lambda_l2: float
    lambda_l1: float
    colsample_bytree: float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_raw_data: Path
    model_path: Path
    all_params: dict
    metric_file_path: Path
    preprocessor_path: Path
    target_column: str


@dataclass(frozen=True)
class ModelMonitoringConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_path: Path
    preprocessor_path: Path