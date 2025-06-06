import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.exception import CustomException
from src.logger.logging import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):

        try:
            num_features = ['age','study_hours_per_day','social_media_hours','netflix_hours','attendance_percentage','sleep_hours','exercise_frequency','mental_health_rating']
            cat_features = ['gender','part_time_job','diet_quality','parental_education_level','internet_quality','extracurricular_participation']
            ohe_features = ['gender','part_time_job','extracurricular_participation']
            ordinal_enc_features = ['diet_quality','parental_education_level','internet_quality']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent'))
                ]
            )

            ohe_pipeline = Pipeline(
                steps=[
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            ordinal_enc_pipeline = Pipeline(
                steps=[
                    ('ordinalencoder', OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=-1)),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor =ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features),
                    ('ohe_pipeline', ohe_pipeline, ohe_features),
                    ('ordinal_enc_pipeline', ordinal_enc_pipeline, ordinal_enc_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation initiated")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")

            train_df = train_df.drop(columns=['student_id'], axis=1)
            test_df = test_df.drop(columns=['student_id'], axis=1)
            logging.info("Dropped student_id column from train and test dataframes")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj()

            target_col_name = 'exam_score'
            # num_features = ['age','study_hours_per_day','social_media_hours','netflix_hours','attendance_percentage','sleep_hours','exercise_frequency','mental_health_rating']

            input_features_train = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_features_test = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying preprocessing object on traning and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing completed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except CustomException as e:
            raise CustomException(e, sys)
        