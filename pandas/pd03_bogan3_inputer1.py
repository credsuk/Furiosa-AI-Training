import pandas as pd
import numpy as np

data = pd.DataFrame([
    [2, np.nan, 6, 8, 10],
    [2, 4, np.nan, 8, np.nan],
    [2, 4, 6, 8, 10],
    [np.nan, 4, np.nan, 8, np.nan],
]).T

# 컬럼명 넣기
data.columns = ['x1', 'x2', 'x3', 'x4']

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

imputer = SimpleImputer() # 디폴트 평균값
data2 = imputer.fit_transform(data)
print(data2)

print("==============================")
imputer2 = SimpleImputer(strategy='mean') # 평균값 / 디폴트와 동일함
data3 = imputer2.fit_transform(data)
print(data3)

print("==============================")
imputer3 = SimpleImputer(strategy='median') # 중위값
data4 = imputer3 = SimpleImputer(strategy='median') # 중위값
imputer3.fit_transform(data)
print(data4)
