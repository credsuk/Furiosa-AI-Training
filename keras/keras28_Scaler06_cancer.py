# keras21_sigmoid_matrics_cancer.py 불러옴

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer # load_breast_cancer 유방암관련 데이터셋 / 분류형 데이터 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_breast_cancer()

x = datasets["data"]
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3,
    random_state=124, 
    stratify=y, # y데이터를 stratify해라 골고루 나오게 해라
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
    patience=500,
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
print("================= keras28_Scaler06_cancer =================")
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
이전 값
훈련에 걸린시간 : 23.94 초
loss : [0.17593811452388763, 0.9298245906829834]
rmse : 0.23067606985569

훈련에 걸린시간 : 19.19 초
loss : 0.3699551224708557
acc : 0.9181
acc_score : 0.9181
"""

"""
이후 값 MinMaxScaler
Epoch 224/10000
훈련에 걸린시간 : 14.05 초
loss : 0.159442737698555
acc : 0.9357
acc_score : 0.9357
"""

# StandardScaler
# Epoch 515/10000
# 훈련에 걸린시간 : 32.88 초
# loss : 0.14123688638210297
# acc : 0.9474
# acc_score : 0.9474


# MaxAbsScaler
# 훈련에 걸린시간 : 40.62 초
# loss : 0.1220901682972908
# acc : 0.9474
# acc_score : 0.9474


# RobustScaler
# 훈련에 걸린시간 : 35.97 초
# loss : 0.1436488777399063
# acc : 0.9415
# acc_score : 0.9415