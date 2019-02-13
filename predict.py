from preprocessing import preprocess
from joblib import dump, load
import pandas as pd
import numpy as np

#load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
store = pd.read_csv("data/store.csv")
df_train = preprocess(train, store)
df_test = preprocess(test, store)
X_test = df_test.drop(['Id'],axis = 1)

#load model
model_randomForest = load("model/randomForest.joblib")
model_XGboost = load("model/XGboost.joblib")

#make prediciton
randomForest_predict = model_randomForest.predict(X_test)
XGboost_predict = model_XGboost.predict(X_test)

#save result
np.savetxt('result/RF_predict.txt', randomForest_predict, delimiter=',')
np.savetxt('result/XGboost_predict.txt', XGboost_predict, delimiter=',')