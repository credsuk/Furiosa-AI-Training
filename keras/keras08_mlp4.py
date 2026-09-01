import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))

y = np.array([
    [1,2,3,4,5,6,7,8,9,10], 
    [10,9,8,7,6,5,4,3,2,1],
    [9,8,7,6,5,4,3,2,1,0]
]).T

print(x.shape, y.shape) # (10,) (10, 3)

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=350, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

results = model.predict(np.array([10]))
print("results : ", results)

# x보다 y값이 더 클때 모델링을 하는 방법
# 11, 0, -1 소수점 2자리

# loss :  0.002129645785316825
# results :  [[11.000083    0.08854675 -1.0324866 ]]


