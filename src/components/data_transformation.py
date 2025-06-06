import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder

from src.exception import CustomException
from src.logger.logging import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):

        try:
            num_features = ['age','study_hours_per_day','social_media_hours','netflix_hours','attendance_percentage','sleep_hours','exercise_frequency','mental_health_rating','exam_score']
            cat_features = ['gender','part_time_job','diet_quality','parental_education_level','internet_quality','extracurricular_participation']
            ohe_features = ['gender']
            label_enc_features = ['part_time_job','extracurricular_participation']
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
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            label_enc_pipeline = Pipeline(
                steps=[
                    ('labelencoder', LabelEncoder())
                ]
            )

            ordinal_enc_pipeline = Pipeline(
                steps=[
                    ('ordinalencoder', OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value'))
                ]
            )

            preprocessor =ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features),
                    ('ohe_pipeline', ohe_pipeline, ohe_features),
                    ('label_enc_pipeline', label_enc_pipeline, label_enc_features),
                    ('ordinal_enc_pipeline', ordinal_enc_pipeline, ordinal_enc_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        