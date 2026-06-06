"""
End to end training script for the dev salary pred model

Outputs:
1. Saved pipeline
2. Cleaned dataset
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,TargetEncoder
import joblib

from xgboost import XGBRegressor

from preprocessing import load_and_clean, get_feature_columns, TARGET,LOG_TARGET
from evaluation import evaluate_model, plot_prediction, print_observations

# Adding /src to parh so we can import modules
sys.path.insert(0, os.path.dirname(__file__))

#Configuration
RAW_DATA_PATH = 'data/raw/developer-salary-survey.csv'
PROCESSED_DATA = 'data/cleaned/processed_data.csv'
MODEL_OUTPUT_PATH = 'models/salary_pipeline_V2.pkl'
 
RANDOM_STATE = 42
TEST_SIZE = 0.2
 
XGBOOST_PARAMS = {
    'n_estimators': 600,
    'max_depth': 6,
    'learning_rate': 0.03,
    'random_state': RANDOM_STATE,
    'verbosity': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist', # makes execution fast for large datasets.
    'reg_alpha' : 0.05, #L1
    'reg_lambda' :1.0 #L2
}

# Creating the pipelines
def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """
    Build andreturn thescikit-learncolumn transformer

    Numeric Pipeline:
        1. Simple Imputer - fill NaN with median (median filling is more robust to outliers)
        2. Standard Scaler - centre and scale

    Categorical pipeline:
        1. Simple Imputer - fill Nan with the most frequent value.
        2. OneHotEncoder - Convert categories into binary columns, handle unknown = 'ignore'
        handle unkown- makes unseen categories become 0s
    """

    numeric_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',TargetEncoder(smooth ='auto',target_type='continuous', random_state=RANDOM_STATE))
    ])

    preprocessor=ColumnTransformer(transformers=[
    ('num',numeric_pipeline,num_cols),
    ('cat', categorical_pipeline,cat_cols)
    ], remainder ='drop')

    return preprocessor

def build_pipeline(cat_cols:list, num_cols:list) -> Pipeline:
    """
    combine preprocessor + model into one sklearn pipeline
    """
    preprocessor = build_preprocessor(cat_cols,num_cols)
    model =XGBRegressor(**XGBOOST_PARAMS)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model',model)
    ])

    return pipeline
def main():
    print("Developer Salary prediction - training")

    # 1. Load and clean data
    df = load_and_clean(RAW_DATA_PATH)

    # Save processed data
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"Processed data saved to: {PROCESSED_DATA}\n\n")

    # 2. Split features and targets
    X = df.drop(columns=[LOG_TARGET])
    y = df[LOG_TARGET]

    cat_cols,num_cols = get_feature_columns(df)
    print(f"Numeric features: {num_cols}")
    print(f"Categorical features: {cat_cols}\n")

    # 3. Train/test split
    X_train, X_test, y_train,y_test = train_test_split(X,y
                                                       ,test_size=TEST_SIZE,
                                                       random_state=RANDOM_STATE)
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")

    # 4. Build and train pipelines
    print("Building pipeline ...")
    pipeline = build_pipeline(cat_cols, num_cols)

    print("Training XGBoost model ...")
    pipeline.fit(X_train, y_train)
    print("Training complete. \n")

    # 5. Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_metrics = evaluate_model(y_train,y_pred_train, title="training test perfomance")
    test_metrics = evaluate_model(y_test, y_pred_test, title="Test set perfomance")

    plot_prediction(y_test.values,y_pred_test,save_path='data/predictions_plot.png')

    # 6. Save the pipeline
    os.makedirs('../models',exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\n Model saved to: {MODEL_OUTPUT_PATH}")

    # example prediction
    print("\n Sample prediction: \n")

    sample=pd.DataFrame([{
        'Country':"Ukraine",
        "YearsCode":10.0,
        'EdLevel':"Bachelor's",
        'Employment':"Full-time",
        'LanguageHaveWorkedWith':4
    }])
    pred = pipeline.predict(sample)[0]
    print(f"input:{sample.to_dict(orient='records')[0]}")

    mae = test_metrics['mae']
    print(f"Predicted salary: ${pred:,.0f}+/-${mae}")

    print("\n Training script complete. \n")

if __name__ == '__main__':
    main()





    
