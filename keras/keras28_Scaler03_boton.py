"""
keras20_EarlyStopping3_boton.py 복사
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 500,
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs = 70000, 
                 batch_size = 24, 
                 validation_split = 0.2,
                 callbacks = [es],
                )
end_time = time.time()

#4. 평가 예측
print("================= keras28_Scaler03_boton ================")
print("훈련 시간 :", round(end_time - start_time, 2), "초")
loss = model.evaluate(x_test, y_test) # 로스율을 계산
print("loss(mse) : ", loss)

# 평가 지표, 예측
from sklearn.metrics import r2_score, mean_squared_error 
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("R2 :", r2)

mse = mean_squared_error(y_test, y_predict) # 스키드런은 원값과 예측값만 주면된다
print("mse :", mse)


# RMSE함수 정의
def RMSE(y_test, y_predict) : # def로 함수를 정의하고 이름을 뒤에 쓴다 
    return np.sqrt(mean_squared_error(y_test, y_predict)) # sqrt는 루트를 거는 함수


rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"

plt.figure(figsize=(9,6))
plt.plot(hist.history["loss"], c="red", label="loss")
plt.plot(hist.history["val_loss"], c="blue", label="val_loss")

plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("loss")

plt.title("보스턴 loss")

plt.grid()
# plt.show()


"""
이전 값
# loss(mse) :  23.284589767456055
# R2 : 0.7202845708869855

Epoch 7925/70000
loss(mse) :  23.48435401916504
R2 : 0.7178848263322362
mse : 23.484352781936426
RMSE :  4.846065701363987

"""


# 이후 값 MinMaxScaler
"""
Epoch 2398/70000
loss(mse) :  23.36335563659668
R2 : 0.7193383491276265
mse : 23.363355950536878
RMSE :  4.83356555252299

"""

# StandardScaler
# Epoch 1019/70000
# loss(mse) :  23.648204803466797
# R2 : 0.7159165034232793
# mse : 23.64820355600768
# RMSE :  4.862941862289501


# MaxAbsScaler
# 훈련 시간 : 487.61 초
# loss(mse) :  23.259370803833008
# R2 : 0.72058750893292
# mse : 23.259371080928002
# RMSE :  4.82279701842489


# RobustScaler
# 훈련 시간 : 45.23 초
# loss(mse) :  23.38043785095215
# R2 : 0.7191331559287567
# mse : 23.38043702210089
# RMSE :  4.835332152200187
