import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime


def preprocess(df, store):
    df = pd.merge(df, store, on='Store', how='left')
        
    df['Date'] = df['Date'].map(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    # df['DaysSince20150731'] = df['Date'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d") - datetime.strptime("2015-07-31", "%Y-%m-%d")).days)
        
    day2str = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}
    for i in range(1, 8):
        df[day2str[i]] = (df['DayOfWeek'] == i).astype(float)
        
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

    return df


def add_feature(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day  # day in month
    df['WeekOfYear'] = df['Date'].dt.weekofyear
    
    # whether competition is currently open and it started how many months ago
    df['CompetitionOpenMonthsAgo'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpen'] = df['CompetitionOpenMonthsAgo'].apply(lambda x: x if x > 0 else 0)
    
    # whether Promo2 is currently open and it started how many months ago
    df['Promo2OpenMonthsAgo'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
    df['Promo2Open'] = df['Promo2OpenMonthsAgo'].apply(lambda x: x if x > 0 else 0)
    df.loc[df['Promo2SinceYear']==0, 'Promo2Open'] = 0

    # whether PromoInterval is currently open
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    def checkPromoInterval(x):
        mon = month2str[x['Month']]
        return float(x['PromoInterval_'+mon]==1)
    df['IsPromoIntervalMonth'] = df.apply(checkPromoInterval, axis=1)

    df['TueToFri'] = ((df['DayOfWeek'] > 1) & (df['DayOfWeek'] < 6)).astype(float)

    # whether it is currently open and it is currently holiday
    df['OpenHoliday'] = np.where((df['Open'] == 1) & ((df['SchoolHoliday'] == 1)|(df['StateHoliday_0'] != 1)), 2, 1)
    df.loc[df['Open']==0,'OpenHoliday'] = 0
    
    return df


def fill_and_drop(df):
    df['CompetitionDistance'].fillna(100000, inplace=True)
    df['CompetitionOpenMonthsAgo'].fillna(0, inplace=True)
    df['Promo2OpenMonthsAgo'].fillna(0, inplace=True)
    
    df.drop(columns=['Store', 'DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'Date'], inplace=True)
    df.drop(columns=['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceYear', 'Promo2SinceWeek'], inplace=True)

    return df



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