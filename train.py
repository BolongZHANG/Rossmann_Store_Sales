from preprocessing.preprocess import preprocess
from joblib import dump, load
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

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