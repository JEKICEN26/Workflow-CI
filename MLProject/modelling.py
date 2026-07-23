"""
Modelling - MLflow Project (Workflow CI)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import json
import warnings

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    return parser.parse_args()

def main():
    args = parse_args()
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'california_housing_preprocessing.csv')
    df = pd.read_csv(data_path)
    
    feature_columns = [col for col in df.columns if col != 'MedHouseVal']
    X = df[feature_columns]
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)
    
    with mlflow.start_run(run_name="CI_RandomForest") as run:
        mlflow.log_params(vars(args))
        
        model = RandomForestRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")
        
        run_id = run.info.run_id
        print(f"\nRun ID: {run_id}")
        
        with open('run_id.txt', 'w') as f:
            f.write(run_id)

if __name__ == "__main__":
    main()
