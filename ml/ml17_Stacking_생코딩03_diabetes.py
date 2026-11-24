import numpy as np
from sklearn.datasets import load_diabetes
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
x, y = load_diabetes(return_X_y=True)

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


models = [xgb, lg, cat, rf]

train_list = []
test_list = []


#2-1. 모델
for model in models:
    #3-1. 훈련
    model.fit(x_train, y_train, )
    y_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_pred)
    test_list.append(y_test_pred)

    score = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print("{0} ACC : {1:.4f}".format(class_name, score))

# XGBRegressor ACC : 0.3103
# LGBMRegressor ACC : 0.4167
# CatBoostRegressor ACC : 0.4621
# RandomForestRegressor ACC : 0.4582

x_train_new = np.array(train_list).T
x_test_new = np.array(test_list).T
print(x_train_new.shape, x_test_new.shape)

#2-2. 모델
model2 = CatBoostRegressor(verbose=0)

#3-2. 훈련
model2.fit(x_train_new, y_train)

#4. 평가, 예측
y_pred_new = model2.predict(x_test_new)
r2_new = r2_score(y_test, y_pred_new)

print("스태킹 결과 : ", r2_new)
# 스태킹 결과 :  0.39401067251257116
# 훈련결과로 새로운 데이터를 만들어서 결과치를 뽑았다가 스태킹


