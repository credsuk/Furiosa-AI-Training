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
from sklearn.ensemble import StackingRegressor, StackingClassifier



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
lg = LGBMRegressor(verbosity=-1)
cat = CatBoostRegressor(verbose=0)
rf = RandomForestRegressor()

model = StackingRegressor(estimators=[("XGB", xgb), ("LG", lg), ("CAT", cat), ("RF", rf)], final_estimator=CatBoostRegressor(verbose=0))

#3. 훈련
model.fit(x_train, y_train, )

#4. 평가, 예측
y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)

print("스태킹 결과 : ", score)
# 스태킹 결과 :  0.7815913379787133
# 스태킹 결과 :  0.8436915848979505