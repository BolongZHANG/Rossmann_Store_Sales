from preprocessing import preprocess
import pandas as pd
import numpy as np
from joblib import dump, load
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
store = pd.read_csv("data/store.csv")
df_train = preprocess(train, store)
df_test = preprocess(test, store)

Y_train = df_train['Sales']
X_train = df_train.drop(['Sales','Customers'],axis = 1)
X_test = df_test.drop(['Id'],axis = 1)

clf_1 = RandomForestRegressor(max_depth=8, random_state=0, n_estimators=50, oob_score=True)
clf_1.fit(X_train, Y_train)
dump(clf_1, 'randomForest.joblib')

clf_2 = xgb.XGBRegressor(max_depth=8, n_estimators=100, early_stopping_rounds=50)
clf_2.fit(X_train, Y_train)
dump(clf_2, 'XGboost.joblib')

def rmse(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))

    return loss

def cross_validation(clf, X, Y, n):
	
	scorer = make_scorer(rmse)

	scores = cross_val_score(clf, X, Y, cv=n, scoring=scorer)

	print scores

	return scores

