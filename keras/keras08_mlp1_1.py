import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터
# 쉐이프 확인, 데이터 정리
# x = np.array([
#                 [1,2,3,4,5],
#                 [6,7,8,9,10],
# ])

x = np.array([
    [1,6], [2,7], [3,8], [4,9], [5,10]
])
y = np.array([1,2,3,4,5])

print("x.shape : ", x.shape) # (5, 2)
print("y.shape : ", y.shape) # (5,)

# 2. 모델 구성
# input_dim=2는 2차원이라는 뜻이다. 컬럼이 2개, 열이 2개, 픽쳐가 2개 / 컬럼, 열, 픽쳐는 동의어
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1)) # 벡터가 하나라서 1 즉 y의 shape가 (5,)이므로 1이다. 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3)


# 4. 평가, 예측
loss = model.evaluate(x, y) # 그냥 단순히 model의 loss를 보여주는 데이터
print("loss : ", loss)

# loss :  2.612807713242571e-11

results = model.predict(np.array([[6, 11]])) # 데이터를 넣어서 예측값을 보여줌
print("[6, 11]의 예측값 : ", results)  

# loss :  1.2520331993393086e-10
# [6, 11]의 예측값 :  [[5.999976]]


