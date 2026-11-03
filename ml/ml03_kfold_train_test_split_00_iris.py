
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


#2. 모델구성
#
all = all_estimators(type_filter='classifier')  # 분류
# all = all_estimators(type_filter='regressor')  # 회귀

# print("all models : ", all)
# print("sk version : ", sk.__version__) # sk version :  1.9.1
# print("모델의 개수 : ", len(all)) # 분류 모델의 개수 :  44 / 회귀 모델 : 모델의 개수 :  55
# [] > 리스트
# {} > 딕셔너리 
# () > 튜플


for name, model in all:
    # print("name : ", name)
    try:
        model  = model()

        #3. 훈련
        model.fit(x_test, y_test)

        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, "의 정답율 : ", acc)
    except:
        print(name, "모델 오류")


