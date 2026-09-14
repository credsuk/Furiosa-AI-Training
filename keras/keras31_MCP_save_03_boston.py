"""
keras20_EarlyStopping3_boton.py 복사
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
model.add(Dense(6))
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

############### mcp 세이브 파일명 만들기 시작 ####################
mcp_path = "./_save/keras31/boston/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([mcp_path, "k31_", date, "_", filename])
############### mcp 세이브 파일명 만들기 끝 ####################

mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto', 
    save_best_only=True, # 최고의 값만 저장해라
    filepath=filepath, # 파일 경로
    verbose=1, # 보여줄지 안보여줄지 확인하는 항목
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs = 70000, 
                 batch_size = 24, 
                 validation_split = 0.2,
                 callbacks = [es, mcp],
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


# MaxAbsScaler
# 훈련 시간 : 487.61 초
# loss(mse) :  23.259370803833008
# R2 : 0.72058750893292
# mse : 23.259371080928002
# RMSE :  4.82279701842489


# 훈련 시간 : 62.63 초
# loss(mse) :  23.25714683532715
# R2 : 0.7206142122352531
# mse : 23.257148195273604
# RMSE :  4.822566556852648

