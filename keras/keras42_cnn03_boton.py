"""
keras28_Scaler03_boton.py 복사
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape) # (404, 13) (102, 13)


x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(15, (3,1), padding='same', input_shape=(13,1,1), activation='relu'))
model.add(Conv2D(10, (3,1), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(24, activation='relu'))
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
                 epochs = 500, 
                 batch_size = 24, 
                #  validation_split = 0.2,
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



# 이후 값 MinMaxScaler
"""
Epoch 2398/70000
loss(mse) :  23.36335563659668
R2 : 0.7193383491276265
mse : 23.363355950536878
RMSE :  4.83356555252299
"""


# 훈련 시간 : 23.34 초
# loss(mse) :  30.737287521362305
# R2 : 0.6307560361224562
# mse : 30.737288595872784
# RMSE :  5.54412198602022
