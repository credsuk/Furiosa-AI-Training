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

# 이웃과의 뭐가 있다 / # K - Nearest - Naver
# KNN은 원래 모델이다 k=3, k=7i
# 니어 리스트만 가지고 한다
# imputer = KNNImputer(n_neighbors=5) # 디폴트 값이 5
imputer = KNNImputer(n_neighbors=2)
data2 = imputer.fit_transform(data)
print(data2)




