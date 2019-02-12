import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime


def preprocess(df, store):
    df = pd.merge(df, store, on='Store', how='left')

    for i in range(1, 8):
        df['DayOfWeek_'+str(i)] = (df['DayOfWeek'] == i).astype(float)
        
    df['DaysSince20150731'] = df['Date'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d") - datetime.strptime("2015-07-31", "%Y-%m-%d")).days)
        
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    for i in ['0', 'a', 'b', 'c']:
        df['StateHoliday_'+str(i)] = (df['StateHoliday'] == i).astype(float)
        
    for i in ['a', 'b', 'c', 'd']:
        df['StoreType_'+str(i)] = (df['StoreType'] == i).astype(float)
        
    for i in ['a', 'b', 'c']:
        df['Assortment_'+str(i)] = (df['Assortment'] == i).astype(float)
        
    df['PromoInterval'] = df['PromoInterval'].astype(str)
    for mon in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']:
        df['PromoInterval_'+mon] = df['PromoInterval'].map(lambda x: float(mon in x))
        
    df.drop(columns=['DayOfWeek', 'Date', 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'], inplace=True)
    df.fillna(-1, inplace=True)

    return df



if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    store = pd.read_csv("data/store.csv")
    df_train = preprocess(train, store)
    df_test = preprocess(test, store)
    print (df_train.head())