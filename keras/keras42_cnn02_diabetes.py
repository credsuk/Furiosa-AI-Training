"""
keras28_Scaler02_diabetes.py 복사
"""
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=34)

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) # (331, 10)


x_train = x_train.reshape(-1, 10, 1, 1)
x_test = x_test.reshape(-1, 10, 1, 1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(15, (2,1), padding='same', input_shape=(10,1,1), activation='relu'))
model.add(Conv2D(10, (2,1), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 1500,
    restore_best_weights=True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=10000, 
                 batch_size=75, 
                 validation_split=0.2,
                 callbacks=[es],
                )

end_time = time.time()

#4. 평가 예측
print("================= keras28_Scaler02_diabetes ================")
print("훈련시간 : ", round(end_time - start_time, 2), "초")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)


"""
MinMaxScaler
Epoch 1911/10000
loss :  3107.7021484375
r2 : 0.3942459105548386
"""

# 훈련시간 :  288.96 초
# loss :  4308.64697265625
# r2 : 0.16015740198907036