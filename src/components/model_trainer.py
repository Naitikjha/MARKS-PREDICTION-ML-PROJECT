from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfigure:
    trainer_model_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfigure()

    def model_Trainer_Initiate(self,train_array,test_array):
        try:
            logging.info("Split train and test data")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models={ 
                            "LinearRegression":LinearRegression(),
                            "Ridge":Ridge(),
                            "lasso":Lasso(),
                            "K_neighbour":KNeighborsRegressor(),
                            "Decison_classifier":DecisionTreeClassifier(),
                            "Linear_reg":LinearRegression(),
                            "Ada_boost": AdaBoostRegressor(),
                            "Random_forest":RandomForestRegressor()
            
                         } 
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
       
            best_model=models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model for the trainning and test data")

            save_object(
                file_path=self.model_trainer_config.trainer_model_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            R2_score=r2_score(y_test,predicted)

            return R2_score




        except Exception as e:
            raise CustomException(e,sys)
