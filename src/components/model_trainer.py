import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
) 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import customException
from src.logger  import logging
from src.utils import save_object

from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artiacts',"model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        
    def initiate_model_traner(self,train_array, test_array,preprocessor_path):
        try:
            logging.info("spelit trainingand test input data")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "K-Neigbors classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "catBoosting":CatBoostRegressor(verbose=False),
                "AdaBooost": AdaBoostRegressor(),
            }
            
            params = {
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    },
    "Random Forest": {
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Gradient Boosting": {
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Linear Regressor": {},  # fixed
    "XGBClassifier": {
        'learning_rate': [.1, .01, .05, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "catBoosting": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBooost": {
        'learning_rate': [.1, .01, 0.5, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    # Optional: add for K-NN
    "K-Neigbors classifier": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
}

            
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
            
            ## To get  best model score from dict
            best_model_score=max(sorted(model_report.values()))
                
            ## To get best model name from dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise customException("No best model found")
            logging.info(f"Best found model on boh training and testing dataset")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predited=best_model.predict(x_test)
            r2_square=r2_score(y_test,predited)
            return r2_square
        
        except Exception as e:
            raise customException(e,sys)