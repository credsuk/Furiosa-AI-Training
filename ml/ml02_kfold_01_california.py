import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=534) 


#2. 모델구성
# model = RandomForestRegressor()
model = XGBRegressor()


#3. 컴파일, 훈련
score = cross_val_score(model, x, y, cv=kfold) # cv : cross_valdation

#4. 평가, 예측
print("acc : ", score, '\n평균 acc :"', round(np.mean(score), 2))
# acc :  [0.8374588  0.83487797 0.82043537 0.8449454  0.84872904] 
# 평균 acc :" 0.84