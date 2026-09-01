import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) # range(10) : 0~9까지의 숫자를 만들어준다
print(x) # [0 1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 10)) # range(1, 10) : 1~9까지의 숫자를 만들어준다
print(x) # [1 2 3 4 5 6 7 8 9]

x = np.array([range(10), range(21, 31), range(201, 211)]).T # (3, 10) 3행 10열, T함수는 한국말로 전치 라고함
print("x.shape : ", x.shape) # (3, 10) -> (10, 3) 

y = np.array(range(1, 11)) # range(1, 11) : 1~10까지의 숫자
print("y.shape : ", y.shape) # (10,)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=5)


#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

results = model.predict(np.array([[10, 31, 211]]))
print("[10, 31, 211]의 예측값 : ", results)

# loss :  6.184563972055912e-11
# [10, 31, 211]의 예측값 :  [[11.000004]]



