# keras42_cnn01_californaia 복사

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, LSTM, Embedding, GRU, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import numpy as np
import time


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


# print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) # 이렇게도 사용가능
x_test = scaler.transform(x_test) # 같은 비율로 테스트 데이터를 변환한다



# print(x_train.shape, x_test.shape) # (15480, 8) (5160, 8)
x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)
print(x_train.shape, x_test.shape) # (15480, 8, 1) (5160, 8, 1)


#2. 모델구성
model = Sequential()
model.add(GRU(units=10, input_shape=(8, 1))) 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

print(x_train.shape, y_train.shape)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[es],)
end_time = time.time()

#4. 평가 예측
print("================= keras28_Scaler01_californaia ================")
loss = model.evaluate(x_test, y_test)
print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss : ", loss)


# cnn > dnn
# 훈련에 걸린시간 : 424.39 초
# loss :  0.40225476026535034


# 훈련에 걸린시간 : 508.75 초
# loss :  [0.2755866050720215, 0.0027131782844662666]