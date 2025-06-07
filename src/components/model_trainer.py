import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger.logging import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                'Linear Regression':LinearRegression(),
                'Ridge Regression':Ridge(),
                'Lasso Regression':Lasso(),
                'ElasticNet Regression':ElasticNet(),
                'Support Vector Regression':SVR(),
                'Decision Tree Regressor':DecisionTreeRegressor(),
                'Random Forest Regressor':RandomForestRegressor(),
                'Gradient Boosting Regressor':GradientBoostingRegressor(),
                'AdaBoost Regressor':AdaBoostRegressor(),
                'K-Neighbors Regressor':KNeighborsRegressor(),
                'XGBoost Regressor':XGBRegressor(),
                'CatBoost Regressor':CatBoostRegressor(verbose=False)
            }

            params={
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64]
                },
                
                "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64]
                },
                
                "Linear Regression":{},
                
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64]
                },
                
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64]
                },
                
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,10,15,20],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['auto','ball_tree','kd_tree','brute']
                },

                "CatBoost Regressor": {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8]
                },
                
                "Ridge Regression": {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },
                
                "Lasso Regression": {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'selection': ['cyclic', 'random']
                },
                
                "ElasticNet Regression": {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                },
                
                "Support Vector Regression": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                
                "XGBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64],
                    'max_depth': [3, 5, 7]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test, models=models, params=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r_sqr_score = r2_score(y_test, predicted)
            logging.info(f"Best model found: {best_model_name} with R2 Score: {r_sqr_score}")

            return r_sqr_score
        
        except Exception as e:
            raise CustomException(e, sys)