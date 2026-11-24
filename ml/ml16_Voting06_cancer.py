import numpy as np
from sklearn.datasets import load_breast_cancer
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
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    random_state=333, 
    stratify=y, # y데이터를 stratify해라 골고루 나오게 해라
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
xgb = XGBClassifier()
lg = LGBMClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('LG', lg), ('CAT',cat)], #평가자 / 
    voting='hard', # 디폴트 hard
    # voting='soft',
    # verbose=1,
)


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 : ", result)

y_pred = model.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
print("acc : ", acc_score)


# ================= RandomForestRegressor ================
# acc :  0.986013986013986


# ================= Voting Hard ================
# acc :  0.993006993006993


# ================= Voting Soft ================
# acc :  0.986013986013986