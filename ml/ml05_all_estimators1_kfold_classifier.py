
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings("ignore") # warnings은 무시하고 넘어가라

#1. 데이터
x, y = load_iris(return_X_y=True)
# print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=21332, stratify=y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=333) # n_splits 만큼 나눠준다 / 지금은 5등분


all = all_estimators()
for name, model in all:
    try:
        model = model()
        score = cross_val_score(model, x, y, cv=kfold)
        print(name, "의 평균 정답율 : ", round(np.mean(score), 2))
    except:
        print(name, "모델 오류")

