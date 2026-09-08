import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer # load_breast_cancer 유방암관련 데이터셋 / 분류형 데이터 


#1. 데이터
datasets = load_breast_cancer()

# print(datasets.DESCR) # 디스크라이브 묘사하다의 약자
# 판다스에도 있다
"""
Breast cancer Wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

:Number of Instances: 569

:Number of Attributes: 30 numeric, predictive attributes and the class

:Attribute Information:
    - radius (mean of distances from center to points on the perimeter)
    - texture (standard deviation of gray-scale values)
    - perimeter
    - area
    - smoothness (local variation in radius lengths)
    - compactness (perimeter^2 / area - 1.0)
    - concavity (severity of concave portions of the contour)
    - concave points (number of concave portions of the contour)
    - symmetry
    - fractal dimension ("coastline approximation" - 1)
    ......
"""

# 이것도 판다스에 존재한다
# print(datasets.feature_names) # 컬럼 확인
"""
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
"""
# 실무에서는 무조건 판다스


x = datasets["data"]
y = datasets.target

# print(type(x)) # 판다스 자체도 넘파이로 이루어진 함수의 집합체 이다 / <class 'numpy.ndarray'>
# print(y)

# 뭔가 특이한게 있는지 체크하는 함수 / 범주형 데이터를 받으면 꼭 해봐야 한다
# print(np.unique(y)) # [0 1]

# 각 라벨별로 몇개가 존재하는지 확인하는 방법
# print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357]))
# 이정도면 아주 정상적인 범위

# pandas를 이용해서 / 0과 1의 개수가 몇개인지 찾아보기 / 좀 더 가독성이 있게 나옴
# print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212


# print(pd.Series(y).value_counts())
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3,
    random_state=124, 
    stratify=y, # y데이터를 stratify해라 골고루 나오게 해라
)
# 찍어보니 차이가 생길 수 있다 그래서 문제가 생길 수 있다
# stratify가 분류 모델에서는 꼭 필요하다
# print(np.unique(y_train, return_counts=True)) # stratify 전 : (array([0, 1]), array([155, 271])) | stratify 후 : (array([0, 1]), array([148, 250]))
# print(np.unique(y_test, return_counts=True)) # stratify 전 : (array([0, 1]), array([57, 86])) | stratify 후 : (array([0, 1]), array([ 64, 107]))

# print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
# print(y_train.shape, y_test.shape) # (398,) (171,)


#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=30, activation='relu'))
model.add(Dense(29, activation='relu'))
model.add(Dense(41, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 무조건 sigmoid를 넣는다 이거 하나다
# 무조건인 놈들이 있다 sigmoid는 무조건
# 시그모이드는 계산만 시키지 반올림 시키지 않는다 왜냐하면 웨이트값을 조절해야 하기 때문임


#3. 컴파일, 훈련
# mse는 회귀모델이니깐 2진 분류에서는 무조건 binary_crossentropy를 사용해야함
model.compile(loss='binary_crossentropy', optimizer='adam', 
            #   metrics=['accuracy'], # 보조지표로 정확도 accuracy 가 나옴
              metrics=['acc'], # 이렇게 줄여서도 사용 가능함 리스트니깐 나중에 친구들이 나옴, 이 앖이 0과 1로 반올림 해줌
            )

es = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=200,
    restore_best_weights=True
)

start_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=10000,
          batch_size=46,
          validation_split=0.2,
          callbacks=[es]
        )
end_time = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
# print(np.round(y_pred[:10]))

rmse = root_mean_squared_error(y_test, y_pred)

print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss :", loss[0])
print("acc :", round(loss[1], 4))
# print("rmse :", rmse)

from sklearn.metrics import accuracy_score # accuracy 점수를 확인
acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", np.round(acc_score, 4))


"""
훈련에 걸린시간 : 23.94 초
loss : [0.17593811452388763, 0.9298245906829834]
rmse : 0.23067606985569

훈련에 걸린시간 : 19.19 초
loss : 0.3699551224708557
acc : 0.9181
acc_score : 0.9181
"""

