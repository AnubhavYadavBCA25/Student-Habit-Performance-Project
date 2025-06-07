import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_pth = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_pth)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(
        self,
        age: int,
        gender: str,
        study_hours_per_day: float,
        social_media_hours: float,
        netflix_hours: float,
        part_time_job: str,
        attendance_percentage: float,
        sleep_hours: float,
        diet_quality: str,
        exercise_frequency: int,
        parental_education_level: str,
        internet_quality: str,
        mental_health_rating: int,
        extracurricular_participation: str
    ):
        self.age = age
        self.gender = gender
        self.study_hours_per_day = study_hours_per_day
        self.social_media_hours = social_media_hours
        self.netflix_hours = netflix_hours
        self.part_time_job = part_time_job
        self.attendance_percentage = attendance_percentage
        self.sleep_hours = sleep_hours
        self.diet_quality = diet_quality
        self.exercise_frequency = exercise_frequency
        self.parental_education_level = parental_education_level
        self.internet_quality = internet_quality
        self.mental_health_rating = mental_health_rating
        self.extracurricular_participation = extracurricular_participation

    def get_data_as_df(self):
        try:
            data_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "study_hours_per_day": [self.study_hours_per_day],
                "social_media_hours": [self.social_media_hours],
                "netflix_hours": [self.netflix_hours],
                "part_time_job": [self.part_time_job],
                "attendance_percentage": [self.attendance_percentage],
                "sleep_hours": [self.sleep_hours],
                "diet_quality": [self.diet_quality],
                "exercise_frequency": [self.exercise_frequency],
                "parental_education_level": [self.parental_education_level],
                "internet_quality": [self.internet_quality],
                "mental_health_rating": [self.mental_health_rating],
                "extracurricular_participation": [self.extracurricular_participation]
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)