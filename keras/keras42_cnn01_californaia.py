
"""
keras28_Scaler01_californaia.py 복사 해옴

"""
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) # 이렇게도 사용가능
x_test = scaler.transform(x_test) # 같은 비율로 테스트 데이터를 변환한다

x_train = x_train.reshape(-1, 8, 1, 1)
x_test = x_test.reshape(-1, 8, 1, 1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(15, (2,1), input_shape=(8,1,1), activation='relu'))
model.add(Conv2D(10, (2,1), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

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


# MinMaxScaler
# 훈련에 걸린시간 :  171.99 초
# loss :  0.3200035095214844


# cnn > dnn
# 훈련에 걸린시간 : 424.39 초
# loss :  0.40225476026535034