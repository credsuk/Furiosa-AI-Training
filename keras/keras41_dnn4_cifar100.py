# keras39_MaxPooling4_cifar100 복사

from tensorflow.keras.datasets import cifar100

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
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
# x_train = x_train.reshape(-1, 32, 32, 3)
# x_test = x_test.reshape(-1, 32, 32, 3)
# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# x 데이터 4차원 데이터로 변경
x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)


# OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))
# print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)


#2. 모델 구성
model = Sequential()
model.add(Dense(213, input_shape=(3072,), activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(324, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(244, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(184, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(94, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(78, activation='relu'))
model.add(Dense(100, activation='softmax')) # (100, )

model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=200,
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


# Flatten
# 걸린시간 :  218.95 초
# loss :  2.4459893703460693
# acc :  0.3792000114917755
# acc_score : 0.3792


# GlobalAveragePooling2D
# 걸린시간 :  910.01 초
# loss :  2.287480354309082
# acc :  0.4171999990940094
# acc_score : 0.4172


# Dense
# 걸린시간 :  44.2 초
# loss :  3.3143632411956787
# acc :  0.22310000658035278
# acc_score : 0.2231