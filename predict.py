from preprocessing import preprocess
from joblib import dump, load
import pandas as pd
import numpy as np
from train import rmspe
from sklearn.metrics import mean_squared_error



if __name__ == "__main__":
    #load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    store = pd.read_csv("data/store.csv")
    
    df_train = preprocess(train, store)
    df_test = preprocess(test, store)
    
    X_train = df_train.drop(['Store', 'Sales','Customers'], axis=1)
    y_train = df_train['Sales']
    X_test = df_test.drop(['Store', 'Id'], axis=1)

    #load model
    model_randomForest = load("model/randomForest.joblib")
    model_XGboost = load("model/XGboost.joblib")

    #make prediciton
    randomForest_predict = model_randomForest.predict(X_train)
    XGboost_predict = model_XGboost.predict(X_train)

    # print ("Verify rmspe:")
    # print (rmspe(y_train, randomForest_predict))
    # res_rf = pd.DataFrame(list(zip(y_train, randomForest_predict)), columns=['true', 'pred'])
    # res_rf = res_rf[res_rf['true']!=0]
    # print (np.sqrt(mean_squared_error(res_rf['true']/res_rf['true'], res_rf['pred']/res_rf['true'])))
    
    # print (rmspe(y_train, XGboost_predict))
    # res_xgb = pd.DataFrame(list(zip(y_train, XGboost_predict)), columns=['true', 'pred']) 
    # res_xgb = res_xgb[res_xgb['true']!=0]
    # print (np.sqrt(mean_squared_error(res_xgb['true']/res_xgb['true'], res_xgb['pred']/res_xgb['true'])))


    # #save result
    # np.savetxt('result/RF_predict.txt', randomForest_predict, delimiter=',')
    # np.savetxt('result/XGboost_predict.txt', XGboost_predict, delimiter=',')