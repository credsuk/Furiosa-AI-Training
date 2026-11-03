"""
keras64_LSTM05_kaggle_bike.py 복사
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.shape, test_csv.shape, submission_csv.shape) # (10886, 11) (6493, 8) (6493, 2)

x = train_csv.drop(["casual", "registered", "count"], axis=1)
y = train_csv['count']
# print(x.shape, y.shape) # (10886, 8) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5437089)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)




# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=333) 


#2. 모델구성
model = XGBRegressor()


#3. 컴파일, 훈련
score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross_valdation

#4. 평가, 예측
print("rc : ", score, '\n평균 rc :"', round(np.mean(score), 2))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse :", rmse)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

# rmse : 164.5293061376149
# r2 : 0.16519790887832642


