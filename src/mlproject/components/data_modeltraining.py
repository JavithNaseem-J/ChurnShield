import pandas as pd
import os
from mlproject import logger
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from mlproject.entities.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        classifier = HistGradientBoostingClassifier(l2_regularization=self.config.l2_regularization, max_depth=self.config.max_depth,max_iter=self.config.max_iter, random_state=42)
        classifier.fit(train_x, train_y)

        joblib.dump(classifier, os.path.join(self.config.root_dir, self.config.model_name))

