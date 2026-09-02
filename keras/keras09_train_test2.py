import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 학습용 데이터 70%, 검증용 데이터 30%
# - 무식하게 나누기
# x_train = np.array([1,2,3,4,5,6,7]) # train은 학습용이라는 정의
# y_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

# - 함수로 나누기
x_train = x[0:7]    # 이렇게 하면 나누어 진다
y_train = y[:7]     # 0은 생략가능
x_test = x[7:10]     # 시작부터 ~ 끝까지
y_test = y[7:]      # 제일 끝도 명시할 필요 없음

# - 내가 원했던걸 제미나이로 찾음 
# x_split = np.split(x, [7])
# x_train = x_split[0]
# x_test = x_split[1]

print(x_train)
print(x_test)


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



