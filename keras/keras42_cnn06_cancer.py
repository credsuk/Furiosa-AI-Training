# keras21_sigmoid_matrics_cancer.py 불러옴

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
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

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (398, 30) (171, 30)


x_train = x_train.reshape(-1, 10, 3, 1)
x_test = x_test.reshape(-1, 10, 3, 1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(15, (2,1), padding='same', input_shape=(10,3,1), activation='relu'))
model.add(Conv2D(10, (2,1), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 무조건 sigmoid를 넣는다 이거 하나다




#3. 컴파일, 훈련
# mse는 회귀모델이니깐 2진 분류에서는 무조건 binary_crossentropy를 사용해야함
model.compile(loss='binary_crossentropy', optimizer='adam', 
            #   metrics=['accuracy'], # 보조지표로 정확도 accuracy 가 나옴
              metrics=['acc'], # 이렇게 줄여서도 사용 가능함 리스트니깐 나중에 친구들이 나옴, 이 앖이 0과 1로 반올림 해줌
            )

es = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=10,
    restore_best_weights=True
)

start_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=500,
          batch_size=46,
          # validation_split=0.2,
          # callbacks=[es]
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
이후 값 MinMaxScaler
Epoch 224/10000
훈련에 걸린시간 : 14.05 초
loss : 0.159442737698555
acc : 0.9357
acc_score : 0.9357
"""

# 훈련에 걸린시간 : 26.52 초
# loss : 0.1719151884317398
# acc : 0.9357
# acc_score : 0.9357