import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([
    range(10), range(21, 31), range(201, 211)
]).T

y = np.array([
    [1,2,3,4,5,6,7,8,9,10], 
    [10,9,8,7,6,5,4,3,2,1]
]).T

print("x.shape : ", x.shape) # (10, 3)
print("y.shape : ", y.shape) # (10, 2)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(2))
model.add(Dense(2))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

results = model.predict(np.array([[11, 31, 211]]))
print("[11, 31, 211]의 예측값 : ", results)


