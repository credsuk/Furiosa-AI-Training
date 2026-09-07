from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] train_test_split로 잘라요!!

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=123)

print(x_train.shape, x_val.shape, x_test.shape) # (8,) (4,) (4,)


#2. 모델구축
model = Sequential()
model.add(Dense(1, input_dim = 1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 10, batch_size = 10, verbose = 1, validation_data = (x_val, y_val))


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

