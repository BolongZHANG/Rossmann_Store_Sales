from preprocessing import data_prepare
import pandas as pd
import numpy as np
from joblib import dump, load
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


def rmspe(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred*0.995)
    res = pd.DataFrame(list(zip(y_true, y_pred)), columns=['true', 'pred'])
    res = res[res['true']!=0]
    loss = np.sqrt(np.mean(np.square((res['true'] - res['pred']) / res['true']), axis=0))
    return loss


def cross_validation(clf, X, y, n):
    y = np.log1p(y)
    scorer = make_scorer(rmspe)
    scores = cross_val_score(clf, X, y, cv=n, scoring=scorer)
    return scores


if __name__ == "__main__":

    X_train, y_train, _ = data_prepare(True)

    # rf = RandomForestRegressor(n_estimators=15, criterion='mse', max_depth=15, n_jobs=3)
    # scores = cross_validation(rf, X_train, y_train, 5)
    # print (scores)
    # print (np.mean(scores))
    
    xgb = XGBRegressor(max_depth=12, n_estimators=1500, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, early_stopping_rounds=100, n_jobs=3)
    # scores = cross_validation(xgb, X_train, y_train, 5)
    # print (scores)
    # print (np.mean(scores))
    
    xgb.fit(X_train, y_train)
    dump(xgb, 'model/XGboost.joblib')