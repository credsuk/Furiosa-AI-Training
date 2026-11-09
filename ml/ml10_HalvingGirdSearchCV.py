# Halving 반땡 그리드 써치
import numpy as np
from sklearn.datasets import load_diabetes, load_iris, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
# from sklearn.experimental import enable_halving_search_cv # 정식 버젼이 아니면 여기 패키지에 이걸 실행해야 정상 동작한다
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=21332, test_size=0.2
)


n_splits=5
kford = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    # 2 x 3 x 3 = 18 candidates
    {'n_estimators': [100, 500], 'max_depth': [6, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]},
    
    # 10.12 -> 10, 12 로 수정 (4 x 3 = 12 candidates)
    {'max_depth': [6, 8, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]},
    
    # leaning_rage -> learning_rate 로 수정 (4 x 3 = 12 candidates)
    {'min_child_weight': [2, 3, 5, 10], 'learning_rate': [0.1, 0.01, 0.001]}
]
# 총 18 + 12 + 12 = 42 candidates (5 folds 기준 총 210 fits)


#2. 모델구성
xgb = XGBClassifier()

# 매핑을 한거다 실제로 모델이 아니라 무식한 알고리즘이다
model = HalvingGridSearchCV(xgb, parameters, cv=kford,
                    verbose=1,
                    refit=True,
                    # n_jobs=-1, # CPU 사용 수(양수) 전부 사용하고 싶으면 -1
                    min_resources=100, # 2 * 5 * 10 / 2그냥 두고 5cv * 클래스갯수
                    max_resources=1437, # x_train 행의 수
                    factor=2, # 디폴트 3
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
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
# model.scroe :  0.9333333333333333
# best_score :  0.9416666666666668
# 걸린시간 :  5.73 초

# RandomizedSearchCV
# 최적의 파라미터 : {'max_depth': 6, 'learning_rate': 0.1}
# model.scroe :  0.9333333333333333
# best_score :  0.9416666666666668
# 걸린시간 :  5.34 초


# HalvingGridSearchCV
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 100}
# model.scroe :  0.9333333333333333
# best_score :  0.9555555555555555
# 걸린시간 :  6.31 초