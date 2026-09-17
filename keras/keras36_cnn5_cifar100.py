from tensorflow.keras.datasets import cifar100

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)

# print(np.unique(y_train, return_counts=True)) 

# MaxAbsScaler 데이터 스케일링
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# x 데이터 4차원 데이터로 변경
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))
# print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
model.add(Dropout(0.2))

model.add(MaxPooling2D())

model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.1))
# model.add(Conv2D(142, (2,2), activation='relu'))
# model.add(Dropout(0.1))
model.add(Conv2D(86, (2,2), activation='relu'))
# model.add(Conv2D(37, (2,2), activation='relu'))
model.add(Flatten())
# model.add(Dense(167, activation='relu'))
model.add(Dense(73, activation='relu'))
model.add(Dense(41, activation='relu'))
model.add(Dense(100, activation='softmax')) # (100, )

# model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

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
    validation_split=0.2,
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
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_predict)
print("acc_score :", acc_score)


# 0.4 acc 기준


# 걸린시간 :  878.62 초
# loss :  2.979702949523926
# acc :  0.29809999465942383
# acc_score : 0.2981


