from preprocessing import data_prepare
from joblib import dump, load
import pandas as pd
import numpy as np
from train import rmspe
from sklearn.metrics import mean_squared_error



if __name__ == "__main__":
    #load data
    _, _, X_test = data_prepare()
    X_test.drop(columns=['Open'], inplace=True)
    test = pd.read_csv("data/test.csv")
    test = test.loc[:, ['Id', 'Open']]

    #load model
    xgb = load("model/XGboost.joblib")

    #make prediciton
    test['Sales'] = xgb.predict(X_test)
    test.loc[test['Open']==0, 'Sales'] = 0
    test.drop(columns=['Open']).to_csv('result/xgb_1500.csv', index=False)