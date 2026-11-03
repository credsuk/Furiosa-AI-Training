# keras64_LSTM01_calfifornia 복사

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import time


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


# print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 이렇게도 사용가능
x_test = scaler.transform(x_test) # 같은 비율로 테스트 데이터를 변환한다



# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=333) 


#2. 모델구성
model = DecisionTreeRegressor()


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

# rmse : 0.8281930167592684
# r2 : 0.4844044140758669