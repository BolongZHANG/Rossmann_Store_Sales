from preprocessing import preprocess, add_feature, fill_and_drop
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
    res = pd.DataFrame(list(zip(y_true, y_pred)), columns=['true', 'pred'])
    res = res[res['true']!=0]
    loss = np.sqrt(np.mean(np.square((res['true'] - res['pred']) / res['true']), axis=0))
    return loss


def cross_validation(clf, X, Y, n):
    scorer = make_scorer(rmspe)
    scores = cross_val_score(clf, X, Y, cv=n, scoring=scorer)
    return scores


if __name__ == "__main__":

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    store = pd.read_csv("data/store.csv")
    
    df_train = preprocess(train, store)
    df_train = add_feature(df_train)
    df_train = fill_and_drop(df_train)
    
    df_test = preprocess(test, store)
    df_test = add_feature(df_test)
    df_test = fill_and_drop(df_test)

    X_train = df_train.drop(['Sales','Customers'], axis=1)
    y_train = df_train['Sales']
    X_test = df_test.drop(['Id'], axis=1)

    # rf = RandomForestRegressor(max_depth=8, random_state=0, n_estimators=50, oob_score=True)
    # rf.fit(X_train, y_train)
    # print (sorted(list(zip(X_train.columns, rf.feature_importances_)), key=lambda x: x[1]))    
    
    xgb = XGBRegressor(max_depth=10, n_estimators=500, learning_rate=0.03, subsample=0.9, colsample_bytree=0.7, early_stopping_rounds=100, n_jobs=3)
    # xgb.fit(X_train, y_train)
    scores = cross_validation(xgb, X_train, y_train, 5)
    print (scores)
    print (np.mean(scores))
    # dump(clf_2, 'model/XGboost.joblib')