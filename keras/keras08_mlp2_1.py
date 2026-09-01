import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([
                [1,2,3,4,5,6,7,8,9,10], 
                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                [9,8,7,6,5,4,3,2,1,0]
            ])

y = np.array([1,2,3,4,5,6,7,8,9,10])

print("x.shape ORG :  ", x.shape) # (3, 10)
x = x.T

print("x.shape Trans : ", x.shape) # (10, 3)
print("y.shape : ", y.shape) # (10,)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

results = model.predict(np.array([[10, 1.3, 0]]))
print("[10, 1.3, 0]의 예측값 : ", results)

# 0.015 이하면 합격
# [10, 1.3, 0]의 예측값 10.00....

# loss :  5.187169335840736e-06
# [10, 1.3, 0]의 예측값 :  [[10.005202]]


