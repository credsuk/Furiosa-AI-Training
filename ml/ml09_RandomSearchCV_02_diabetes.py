import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRFRegressor
import time
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=21332, test_size=0.2
)


n_splits=5
kford = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {'n_estimators': [50, 100, 300, 350, 380, 500], 'max_depth': [6, 10, 12, 15, 17, 23, 25], 'learning_rate': [0.1, 0.01, 0.02, 0.05, 0.001]},
    {'max_depth': [6, 8, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]},
    {'min_child_weight': [2, 3, 5, 10], 'learning_rate': [0.1, 0.01, 0.001]}
]
# 총 18 + 12 + 12 = 42 candidates (5 folds 기준 총 210 fits)


#2. 모델구성
xgb = XGBRFRegressor()

# 매핑을 한거다 실제로 모델이 아니라 무식한 알고리즘이다
model = RandomizedSearchCV(xgb, parameters, cv=kford,
                    verbose=1,
                    n_jobs=-1, # CPU 사용 수(양수) 전부 사용하고 싶으면 -1
                    random_state=213212, # 랜덤값 고정
                    # refit=True, # 다시 훈련시킬까? True가 기본값
                    # error_score='raise'  # 첫 번째 실패 시 즉시 상세 에러를 띄웁니다
                )

#3. 컴파일, 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()


print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)

#4. 평가, 예측
print("model.scroe : ", model.score(x_test, y_test))
print("best_score : ", model.best_score_)

print("걸린시간 : ", round(end - start, 2), "초")


# GridSearchCV
# Fitting 5 folds for each of 234 candidates, totalling 1170 fits
# model.scroe :  0.08533053624844367
# best_score :  0.07773000990517151
# 걸린시간 :  84.08 초


# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# model.scroe :  0.0852331213057329
# best_score :  0.07678506821187185
# 걸린시간 :  10.55 초
