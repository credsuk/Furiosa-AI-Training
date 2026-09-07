from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# 실습 8개, 4개, 4개 잘라봅시다
# 시계열 같은 경우에는 많이 사용한다

x_train = x[:8]
x_val = x[8:12]
x_test = x[12:]

y_train = y[:8]
y_val = y[8:12]
y_test = y[12:]

# print(x_train.shape, x_val.shape, x_test.shape) # (8,) (4,) (4,)


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1, validation_data = (x_val, y_val))
# val_loss를 기준으로 잡고 확인을 해야한다
# val_loss가 계속 떨어지는 중이라면 epochs를 늘린다


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)