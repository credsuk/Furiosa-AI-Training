import numpy as np
from sklearn.datasets import load_diabetes, load_iris, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
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
model = GridSearchCV(xgb, parameters, cv=kford,
                    verbose=1,
                    n_jobs=-1, # CPU 사용 수(양수) 전부 사용하고 싶으면 -1
                    error_score='raise'  # 첫 번째 실패 시 즉시 상세 에러를 띄웁니다
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