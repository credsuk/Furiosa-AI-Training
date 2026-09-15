"""
keras20_EarlyStopping3_boton.py 복사
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time
import datetime


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Dropout(0.3))
model.add(Dense(6))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 50,
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs = 100, 
                 batch_size = 24, 
                 validation_split = 0.2,
                #  callbacks = [es],
                )
end_time = time.time()

#4. 평가 예측
print("================= keras31_MCP_save_03_boston ================")
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


# GPU
# 훈련 시간 : 5.86 초
# loss(mse) :  59.24783706665039
# R2 : 0.2882616664112909
# mse : 59.247838026993435
# RMSE :  7.697261722651337

# CPU
# 훈련 시간 : 7.1 초
# loss(mse) :  46.05801773071289
# R2 : 0.4467096390507944
# mse : 46.05801899432118
# RMSE :  6.786605852288844