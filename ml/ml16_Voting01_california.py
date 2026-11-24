import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # 나무를 앙상블 시킨 모델, Tree계열
from sklearn.ensemble import BaggingRegressor, BaggingClassifier # 
from xgboost import XGBRegressor, XGBClassifier
# pip install lightgbm
from lightgbm import LGBMRegressor, LGBMClassifier
# pip install catboost
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier



#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    random_state=333, 
    # stratify=y, # y데이터를 stratify해라 골고루 나오게 해라
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
xgb = XGBRegressor()
lg = LGBMRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

# Regressor는 hard voting가 없다, 그래서 파라미터가 존재하지 않는다
model = VotingRegressor(
    estimators=[('XGB', xgb), ('RF', rf), ('LG', lg), ('CAT',cat)], #평가자 / 
)


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 : ", result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)


# DecisionTreeRegressor
# R2 :  0.6067773609803491


# BaggingRegressor
# R2 :  0.7726016058735403


# ================= Voting Soft ================
# R2 :  0.841300105846214