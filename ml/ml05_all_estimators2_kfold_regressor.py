
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings("ignore") # warnings은 무시하고 넘어가라


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


# print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 이렇게도 사용가능
x_test = scaler.transform(x_test) # 같은 비율로 테스트 데이터를 변환한다

#2. 모델구성

# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=534) 

all = all_estimators()

for name, model in all:
    try:
        model = model()
        r2 = cross_val_score(model, x, y, cv=kfold)
        print(name, "의 정답율 : ", round(np.mean(r2), 2))
    except:
        print(name, "모델 오류")