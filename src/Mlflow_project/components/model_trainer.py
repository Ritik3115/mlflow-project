import pandas as pd
import os
from Mlflow_project import logger
from catboost import CatBoostRegressor

import joblib
from Mlflow_project.entity.config_entity import ModelTrainerConfig




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


        lr = CatBoostRegressor(n_estimators=self.config.n_estimators, depth=self.config.depth, learning_rate=self.config.learning_rate , subsample = self.config.subsample, l2_leaf_reg=self.config.l2_leaf_reg, random_state=42)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))


