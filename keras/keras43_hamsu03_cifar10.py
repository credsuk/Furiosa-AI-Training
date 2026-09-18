# keras36_cnn5_cifar10 복사

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input
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

# x 데이터 4차원 데이터로 변경
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))
# print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)


#2. 모델 구성
# model = Sequential()
# model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(Dropout(0.1))
# model.add(Conv2D(52, (3,3), activation='relu'))
# model.add(Dropout(0.1))
# model.add(Conv2D(30, (2,2), activation='relu'))
# model.add(Flatten())
# model.add(Dense(24, activation='relu'))
# model.add(Dense(10, activation='softmax')) # (10, )

# model.summary() # 532,420

input1 = Input(shape=(32, 32, 3)) 
conv2D1 = Conv2D(32, (3,3),)(input1)
drop1 = Dropout(0.2)(conv2D1)
conv2D2 = Conv2D(64, (2,2), activation='relu')(drop1)
drop2 = Dropout(0.1)(conv2D2)
conv2D3 = Conv2D(52, (3,3), activation='relu')(drop2)
drop3 = Dropout(0.1)(conv2D3)
conv2D4 = Conv2D(30, (2,2), activation='relu')(drop3)
flatten = Flatten()(conv2D4)
dense1 = Dense(24, activation='relu')(flatten)
output1 = Dense(10, activation='softmax')(dense1)
model = Model(inputs=input1, outputs=output1) 

model.summary() # 532,420


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


# 0.67 acc 기준

# 걸린시간 :  580.59 초
# loss :  1.0096254348754883
# acc :  0.6539000272750854

# 걸린시간 :  494.36 초
# loss :  0.947809100151062
# acc :  0.6725000143051147
# acc_score : 0.6725



# Model 함수
# 걸린시간 :  446.76 초
# loss :  1.043209433555603
# acc :  0.6549999713897705
# acc_score : 0.655
