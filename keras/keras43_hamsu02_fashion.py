# keras36_cnn4_fashion 복사

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import time



#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)

# print(np.unique(y_train, return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],dtype=int64))

# MaxAbsScaler 데이터 스케일링
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# x 데이터 4차원 데이터로 변경
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# print(x_train.shape, x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

# OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))
# print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)


#2. 모델 구성
# model = Sequential()
# model.add(Conv2D(32, (5,5), input_shape=(28, 28, 1)))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.1))
# model.add(Conv2D(62, (2,2), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(41, (3,3), activation='relu'))
# model.add(Dropout(0.1))
# model.add(Conv2D(30, (2,2), activation='relu'))
# model.add(Flatten())
# model.add(Dense(44, activation='relu'))
# model.add(Dense(10, activation='softmax')) # (10, )

# model.summary() #  491,305

input1 = Input(shape=(28, 28, 1)) 
conv2D1 = Conv2D(32, (5,5),)(input1)
drop1 = Dropout(0.2)(conv2D1)
conv2D2 = Conv2D(64, (3,3), activation='relu')(drop1)
drop2 = Dropout(0.2)(conv2D2)
conv2D3 = Conv2D(62, (2,2), activation='relu')(drop2)
drop3 = Dropout(0.3)(conv2D3)
conv2D4 = Conv2D(41, (3,3), activation='relu')(drop3)
drop4 = Dropout(0.1)(conv2D4)
conv2D5 = Conv2D(30, (2,2), activation='relu')(drop4)
flatten = Flatten()(conv2D5)
dense1 = Dense(44, activation='relu')(flatten)
output1 = Dense(10, activation='softmax')(dense1)
model = Model(inputs=input1, outputs=output1) 

model.summary() # 491,305



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
)


start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=5000,
    batch_size=228,
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




# 0.92 acc 기준

# 걸린시간 :  224.09 초
# loss :  0.26256242394447327
# acc :  0.9071999788284302
# acc_score : 0.9072

# Model 함수
# loss :  0.26348263025283813
# acc :  0.9047999978065491
# acc_score : 0.9048
