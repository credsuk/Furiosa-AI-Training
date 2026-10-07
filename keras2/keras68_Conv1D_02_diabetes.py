# keras52_opmizer02_diabetes 복사

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
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

# print(np.min(x_train), np.max(x_train)) # 0.0 1.0
# print(np.min(x_test), np.max(x_test)) # -0.09132350124497224 1.2724968314321925
# print(x_train.shape, y_train.shape) # (331, 10) (331,)

x_train = x_train.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)

# print(x_train.shape, x_test.shape) # (331, 10, 1) (111, 10, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(32, (2,), input_shape=(10,1)))
model.add(Conv1D(64, (2,)))
model.add(Flatten())
model.add(Dense(23))
model.add(Dense(14))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

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
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("================= 02_diabetes ================")
print("훈련시간 : ", round(end_time - start_time, 2), "초")
print("loss : ", loss)
print("r2 :", r2)

# learning_rate 0.01
# 훈련시간 :  175.44 초
# loss :  3036.913818359375
# r2 : 0.4080440063890993


# ================= 02_diabetes ================
# 훈련시간 :  56.9 초
# loss :  4549.6953125
# r2 : 0.11317225314112223