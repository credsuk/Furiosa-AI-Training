import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor # 나무를 앙상블 시킨 모델, Tree계열
from sklearn.ensemble import BaggingRegressor # 

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
# model = DecisionTreeRegressor()

# 래핑하는 친구들 
# Bagging은 뭔가? 앙상블을 섞은거
model = BaggingRegressor(
    RandomForestRegressor(), # Bagging이라는 기법이 이걸 사용하겠다
    n_estimators=100, # epocsh로 생각하자, 멏 번이나 돌랄까? > 10번
    random_state=450,
    # n_jobs= -1, # CPU얼마나 사용할래?
    max_samples=0.8, # 80%의 샘플을 사용하겠다
    # bootstrap=True, # 디폴트 / # 데이터 중복 허용
    bootstrap=False,  # 데이터 중복 허용x
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
# 최종 점수 :  0.6067773609803491
# R2 :  0.6067773609803491


# BaggingRegressor
# 최종 점수 :  0.7726016058735403
# R2 :  0.7726016058735403