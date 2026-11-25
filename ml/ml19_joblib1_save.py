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
# XGBRegressor 파라미터 / 요거 라고 써있는거 튜닝할때 강력한놈
parameters = {
    'n_estimators' : 2000,  # 요거
    'learning_rate' : 0.1,  # 요거
    'max_depth' : 7,        # 요거
    'gamma' : 1,
    'min_child_weight' : 1,
    'subsample' : 1,
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'colsample_bynode' : 1,
    'reg_alpha' : 0,
    'reg_lambda' : 1,
    'random_state' : 333,   # 요거
    # 'verbose' : 0
}


model = XGBRegressor(
    # **parameters,
)
model.set_params(
    **parameters, # 이렇게도 파라미터 전달이 가능
    early_stopping_rounds=20,  # 얼리 스타핑을 줄 수 있다
)

#3. 컴파일, 훈련
model.fit(
    x_train, y_train,
    eval_set = [(x_test, y_test)], # 위에 얼리 스타핑을 사용하려면 필요함
    verbose = 1,
)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 : ", result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)


# 디폴트
# 최종 점수 :  0.8258329975945226
# R2 :  0.8258329975945226

# joblib으로 가중지 저장
import joblib
path = './_save/joblib/'
joblib.dump(model, path + 'ml19_joblib1_save.joblib') # 가중치, 경로


# 최종 점수 :  0.8150232661233965
# R2 :  0.8150232661233965

