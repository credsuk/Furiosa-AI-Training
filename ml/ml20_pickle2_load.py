import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor # 나무를 앙상블 시킨 모델, Tree계열
from sklearn.ensemble import BaggingRegressor 
from xgboost import XGBRegressor

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    random_state=335, 
    # stratify=y, # < 이건 빼야되
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# 가중치 불러오기
import pickle
path = './_save/pickle/'
model = pickle.load(open(path + 'ml20_pickle_save.pickle', 'rb'))


#3. 컴파일, 훈련


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 : ", result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)


# 디폴트
# 최종 점수 :  0.8258329975945226
# R2 :  0.8258329975945226


# 최종 점수 :  0.8150232661233965
# R2 :  0.8150232661233965

