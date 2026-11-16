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

# 최빈값 / 가장 자주 나타나는 값
imputer = SimpleImputer(strategy='most_frequent') 
data2 = imputer.fit_transform(data)
print(data2)

# 특정값(777)로 넣기
imputer2 = SimpleImputer(strategy='constant', fill_value=777)
data3 = imputer2.fit_transform(data)
print(data3)

# 사용자 함수 가능 (람다 함수로)
imputer3 = SimpleImputer(
    strategy=lambda x : np.max(x)
)
data4 = imputer3.fit_transform(data)
print(data4)