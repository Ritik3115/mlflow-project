import os
from Mlflow_project import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from Mlflow_project.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # -------------------------------------------
        # Categorical to Numeric Mapping
        data['sex'] = data['sex'].map({'female': 0, 'male': 1})
        data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
        region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        data['region'] = data['region'].map(region_mapping)

        # -------------------------------------------
        # Feature Engineering
        data['bmi_category'] = pd.cut(
            data['bmi'], 
            bins=[0, 18.5, 25, 30, 40, 50, 60],
            labels=['underweight', 'normal', 'overweight', 'obese', 'obese +', 'obese ++']
        )

        data['age_group'] = pd.cut(
            data['age'], 
            bins=[0, 18, 60, 100],
            labels=['child', 'adult', 'senior citizen']
        )

        data['bmi_category'] = data['bmi_category'].cat.codes
        data['age_group'] = data['age_group'].cat.codes

        data['age_bmi'] = data['age'] * data['bmi']
        data['bmi_smoker'] = data['bmi'] * data['smoker']
        data['age_smoker'] = data['age'] * data['smoker']
        data['children_age_ratio'] = data['children'] / (data['age'] + 1)


        selected_features = ['smoker', 'bmi_smoker', 'age_bmi', 'age', 'age_smoker', 'bmi', 'bmi_category']

        data = data[selected_features + ['charges']]  # Keep the target column too if needed


        # -------------------------------------------
        # Train-Test Split
        train, test = train_test_split(data, test_size=0.20, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
