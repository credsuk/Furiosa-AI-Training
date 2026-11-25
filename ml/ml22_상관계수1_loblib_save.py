import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
print(datasets.feature_names) # 퓨처의 이름이 나옴

# load_iris에서는 2개다 허용해준다
# 둘다 해보면 된다 
x = datasets['data']    # 데이터가 딕셔터리 일때
y = datasets.target     # 데이터가 클래스 형태 (클래스의 객체형태 일때)

# 판다스에 데이터 프레임을 만든다
# 행열행터의 데이터을 넣어서 컬럼 이름으로 지정해서 판다스의 데이터프래임 한마디로 판다스 행열 데이터 형태
df = pd.DataFrame(x, columns=datasets.feature_names)
df['Target'] = y

print(df)

# joblib, 핑클로 세이브하기

import joblib
joblib_path = "./_save/joblib/"
joblib.dump(df, joblib_path + "ml22_상관계수1.joblib")
# joblib.load(data_path + "ml22_상관계수1.joblib")

import pickle
pickle_path = './_save/pickle/'
pickle.dump(df, open(pickle_path +  'ml22_상관계수1.pickle', 'wb'))
# pickle.load(open(data_path +  "ml22_상관계수1.pickle", 'rb'))




