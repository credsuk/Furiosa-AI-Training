import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

data = pd.DataFrame([
    [2, np.nan, 6, 8, 10, 8],
    [2, 4, np.nan, 8, np.nan, 4],
    [2, 4, 6, 8, 10, 12],
    [np.nan, 4, np.nan, 8, np.nan, 8],
]).T

# 컬럼명 넣기
data.columns = ['x1', 'x2', 'x3', 'x4']



from xgboost import XGBRegressor
# 디폴트 epochs는 100
xgb = XGBRegressor(
    max_depth=5,
    leaning_rate=0.1,
    random_state=333,
)

# 4. 모델 : .predict(값을 예측해서), (전혀 다른 모델 사용해야 과적합이 안됨) > 이걸 쉽게 한거임
# 모델을 돌려서 결측치값을 체크한다
# Iterative 반복적인 Imputer 결측치 처리
imputer = IterativeImputer(
    estimator=xgb, # 평가자 / 평가자를 xgb로 하겠다
    max_iter=10,  # 10번 돌리겠
    # tol=0.001, # 디폴트 / 이게 기준값, iter로 돌다가 stop를 하는 기준값
    random_state=333, # 랜덤값
)

data2 = imputer.fit_transform(data)
print(data2)

print("실제 반복 횟수: ", imputer.n_iter_) # 2



