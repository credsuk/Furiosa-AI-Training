"""
keras12_R2_RMSE_01_boston.py 복사
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape) # (404, 13) (404,)


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=700, batch_size=24, validation_split=0.2)


#4. 평가 예측
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


# loss(mse) :  23.284589767456055
# R2 : 0.7202845708869855


"""
# validation_split 추가 후 모니터링 튜닝결과

"""

