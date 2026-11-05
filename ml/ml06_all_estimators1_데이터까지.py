
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk
import time

import warnings
warnings.filterwarnings("ignore") # warnings은 무시하고 넘어가라

#1. 데이터
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [iris, cancer, wine, diabetes]
data_name = [ '아이리스', '캔서', '와인', '디지트']


# KFold 준비
n_splist = 5
kfold = KFold(n_splits=n_splist, shuffle=True, random_state=333) # n_splits 만큼 나눠준다 / 지금은 5등분

#2. 모델구성
all = all_estimators(type_filter='classifier')  # 분류
# all = all_estimators(type_filter='regressor')  # 회귀

start = time.time()
for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=21332, stratify=y, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("===================", data_name[index], "=================")
    model_name =[]
    model_acc = []

    for name, model in all:
        try:
            model  = model()

            #3. 훈련
            scores = cross_val_score(model, x_train, y_train, cv=kfold)

            #4 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

            acc = np.round(accuracy_score(y_test, y_pred), 4)

            model_name.append(name)
            model_acc.append(acc)
            
        except:
            continue

    
    model_name = np.array(model_name)
    model_acc = np.array(model_acc)
    max_index = np.argmax(model_acc)

    print('최고 모델 : ', model_name[max_index], 'ACC : ', model_acc[max_index])

end_time = time.time()
print("걸린시간 : ", round(end_time- start, 2), '초')


