from tensorflow.keras.datasets import cifar10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],dtype=int64))

# MaxAbsScaler 데이터 스케일링
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5


# print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
x_train = np.array(x_train).reshape(-1, 1024, 3)
x_test = np.array(x_test).reshape(-1, 1024, 3)
# print(x_train.shape, x_test.shape) # (50000, 1024, 3) (10000, 1024, 3)
# print(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)
# print(np.unique(y_train))

# exit()

#2. 모델 구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(1024, 3))) 
model.add(Dense(24, activation='relu'))
model.add(Dense(10, activation='softmax')) # (10, )

# model.summary()



#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) 

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
)


start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=5000,
    batch_size=732,
    verbose=1,
    validation_split=0.1,
    callbacks=[es]
)
end_time = time.time()




#4. 평가 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("걸린시간 : ", round(end_time - start_time, 2), "초")
print("================= model.evaluate ===========================")
print("loss : ", loss[0])
print("acc : ", loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_predict)
print("acc_score :", acc_score)


# 걸린시간 :  551.84 초
# loss :  2.05485200881958
# acc :  0.2590999901294708
# acc_score : 0.2591

# 0.67 acc 기준


