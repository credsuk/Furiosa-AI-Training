import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # 나무를 앙상블 시킨 모델, Tree계열
from sklearn.ensemble import BaggingRegressor, BaggingClassifier # 

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    random_state=335, 
    # stratify=y, # < 이건 빼야되
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

model = BaggingClassifier(
    DecisionTreeClassifier(), # Bagging이라는 기법이 이걸 사용하겠다
    n_estimators=100, # epocsh로 생각하자, 멏 번이나 돌랄까? > 10번
    random_state=348,
    # n_jobs= -1, # CPU얼마나 사용할래?
    max_samples=0.7, # 80%의 샘플을 사용하겠다
    bootstrap=True, # 디폴트 / # 데이터 중복 허용
    # bootstrap=False,  # 데이터 중복 허용x
)


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 : ", result)

y_pred = model.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
print("acc : ", acc_score)


# ================= 기존 ================
# acc : 0.9415

# ================= DecisionTree ================
# acc :  0.916083916083916

# ================= RandomForest ================
# acc :  0.986013986013986

# ================= Bagging ================
# acc :  0.965034965034965