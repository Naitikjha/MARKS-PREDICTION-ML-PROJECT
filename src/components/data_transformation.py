from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from src .logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass

class TransformerConfig:
    preprocessor_obj_file_path=os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
        
        def __init__(self):
              self.data_transformation_config=TransformerConfig()

        def get_data_transformer_obj(self):
            try:
                    
              categorical_features=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
              numeric_features=['reading score', 'writing score']

              numeric_pipeline=Pipeline(steps=[
                 ("impute",SimpleImputer(strategy="median")),
                 ("scaler",StandardScaler())

              ])

              categorical_pipeline = Pipeline(steps=[
                                               ("imputer", SimpleImputer(strategy="most_frequent")),
                                               ("ohe_encoding", OneHotEncoder(handle_unknown="ignore"))])


              logging.info("Numeric features standard scaling completed")
              logging.info("Categorical features encodeing completed")
              
              preprocessor = ColumnTransformer([
                                     ("numeric_pipeline", numeric_pipeline, numeric_features),
                                     ("categorical_pipeline", categorical_pipeline, categorical_features)])


              return preprocessor
            
            except Exception as e:
                 raise CustomException(e,sys)
            
        def initiate_data_transformation(self,train_data_path,test_data_path):
             try:
                  train_df=pd.read_csv(train_data_path)
                  test_df=pd.read_csv(test_data_path)

                  logging.info("Reading of training and test data completed")
                  preprocessing_obj=self.get_data_transformer_obj()

                  target_column_name="math score"
                  numeric_features=['reading score', 'writing score']

                  input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
                  target_feature_train_df=train_df[target_column_name]
                 
                  input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                  target_feature_test_df=test_df[target_column_name]
                  
                  logging.info("Appling preprocessing objects on trainning and test dataframe")
                  
                  input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                  input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

                  train_arr=np.c_[
                       input_feature_train_arr,np.array(target_feature_train_df)
                  ]

                  test_arr=np.c_[
                       input_feature_test_arr,np.array(target_feature_test_df)
                  ]

                  logging.info("Saving preprocess object.")

                  
                  save_object(
                       file_path=self.data_transformation_config.preprocessor_obj_file_path,
                       obj=preprocessing_obj
                  )

                  return (
                       train_arr,
                       test_arr,
                       self.data_transformation_config.preprocessor_obj_file_path
                  )
                  
                 
             except Exception as e:
                  raise CustomException(e,sys)
                  
                 