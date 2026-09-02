from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape) # (404, 13) (404,)

#2. 모델구성
model = Sequential()
model.add(Dense(2, input_dim=13))
model.add(Dense(6))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=24)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# loss :  24.14023780822754
