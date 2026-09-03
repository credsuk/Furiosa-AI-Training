# 11-3 카피
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
model.fit(x_train, y_train, epochs=500, batch_size=24)


#4. 평가 예측
loss = model.evaluate(x_test, y_test) # 로스율을 계산
print("loss(mse) : ", loss)

# 평가 지표, 예측
from sklearn.metrics import r2_score, mean_squared_error # metrics에 평가하는 놈들이 모여있다
y_predict = model.predict(x_test) # x_test를 넣어서 예측값을 만든다 텐서플로에서 제공하는게 아니라 예측값이 없어서 만들어준다
r2 = r2_score(y_test, y_predict) # 원값과 예측값이 필요하다 그래야 R2에서 비교가 가능하다
print("R2 :", r2)

mse = mean_squared_error(y_test, y_predict) # 스키드런은 원값과 예측값만 주면된다
print("mse :", mse)


# 파이썬은 인터프리터 언어인데 라인위주의 언어이다 띄어쓰기 매우 중요
# 띄어 쓰기를 하면 윗줄에 종속적이다
# RMSE함수 정의
def RMSE(y_test, y_predict) : # def로 함수를 정의하고 이름을 뒤에 쓴다 
    return np.sqrt(mean_squared_error(y_test, y_predict)) # sqrt는 루트를 거는 함수


rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# loss(mse) :  23.284589767456055
# R2 : 0.7202845708869855




