import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 학습용 데이터 70%, 검증용 데이터 30%
# - 무식하게 나누기
x_train = np.array([1,2,3,4,5,6,7]) # train은 학습용이라는 정의
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2. 모델
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=4)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# 통상적으로 훈련한 데이터의 loss이값이 더 좋다 테스트 한 데이터가 값이 더 나쁘다

# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 2.4014e-04
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step - loss: 8.8339e-04
# loss :  0.0008833851316012442




