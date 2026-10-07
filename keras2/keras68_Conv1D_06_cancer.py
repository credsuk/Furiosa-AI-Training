# keras52_opmizer6_cancer.py 불러옴

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D
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

# print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
x_train = x_train.reshape(-1, 10, 3)
x_test = x_test.reshape(-1, 10, 3)
# print(x_train.shape, x_test.shape) # (398, 10, 3) (171, 10, 3)

#2. 모델구성
model = Sequential()
model.add(Conv1D(36, (2,), input_shape=(10,3)))
model.add(Conv1D(48, (2,)))
model.add(GlobalAveragePooling1D())
model.add(Dense(30, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.002

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'], )

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
import z_evaluate
z_evaluate.evaluate(model, x_test, y_test, start_time, end_time, name="06_cancer")

# loss = model.evaluate(x_test, y_test)

# y_pred = model.predict(x_test)
# y_pred = np.round(y_pred)
# rmse = root_mean_squared_error(y_test, y_pred)

# from sklearn.metrics import accuracy_score # accuracy 점수를 확인
# acc_score = accuracy_score(y_test, y_pred)

# print("================= 06_cancer =================")
# print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
# print("loss :", loss[0])
# print("acc :", round(loss[1], 4))
# print("acc_score :", np.round(acc_score, 4))


# RobustScaler
# 훈련에 걸린시간 : 35.97 초
# loss : 0.1436488777399063
# acc : 0.9415
# acc_score : 0.9415


# learning_rate = 0.002
# 훈련에 걸린시간 : 31.57 초
# loss : 0.19548389315605164
# acc : 0.9415
# acc_score : 0.9415

# =================06_cancer ================
# 훈련 시간 : 38.28 초
# loss : 0.4469497799873352
# acc : 0.8655
# r2 : 0.45972758531570435
# mse : 0.1265273541212082
# rmse2 : 0.3557068407535553