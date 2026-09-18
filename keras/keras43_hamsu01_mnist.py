# keras36_cnn3_mnist 복사
# 
import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)
# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test)) # 255 0
# 255가 나오면 문제없이 우리가 원하는 데로 나온거다


####### 스케일링 1 #######
# MinMax스케일링 원값에서 최소값을 뺀것을 최대값으로 나눈다
# 근데 우리는 이미 최소값이 0인걸 알고, 최대값이 255인걸 안다
# 그러면 MaxAbsScaler를 사용해도되고 최대값을 아니깐 바로 아래와 같이 해도된다
# 이미지라서 가능한거
# x_train = x_train/255. # .(플롯) 을 붙이면 플롯 형태로 간다
# x_test = x_test/255.

# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0


####### 스케일링 2 #######
# 이미지에서 많이 사용하는거 2번째
# -1 ~ 1 사이로 만든다
# 255/2의 값으로 중간값을 먼저 빼고 그 후에 나누기를 하면 
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))  # 1.0 -1.0

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
# print(np.unique(y_train, return_counts=True))

# 4차원 데이터로 변환을 위해서 reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# print(x_train.shape, x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)


# Y 데이터의 원핫
from sklearn.preprocessing import OneHotEncoder # 분류이기 때문에 OneHot를 해야함
ohe = OneHotEncoder(sparse_output=False) # 뭔가 혼동스러운 행열(혼동 행열)을 방지하기 위해서 sparse_output 사용함
y_train = y_train.reshape(-1, 1) # 수치를 알면 60000, 1 이렇게 해도 되지만 모르면 전체라는 의미에서 -1, 1 이렇게 하면된다
y_test = y_test.reshape(-1, 1) # 사실상 -1, 1 이게 디폴트

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)



#2. 모델구성
# model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), activation='relu')) # (23, 23, 32)
# model.add(Conv2D(16, (2,2), activation='relu')) # (22, 22, 6)
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation='relu')) # (21, 21, 16)
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation='relu')) # (20, 20, 16)
# model.add(Conv2D(8, (5,5), activation='relu')) # (16, 16, 8) # 추가
# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=8, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(10, activation='softmax')) # (10, )

# model.summary() # 30,970


input1 = Input(shape=(28, 28, 1)) 
conv2D1 = Conv2D(64, (3,3),)(input1)
conv2D2 = Conv2D(32, (3,3), activation='relu')(conv2D1)
drop1 = Dropout(0.2)(conv2D2)
conv2D3 = Conv2D(32, (2,2), activation='relu')(drop1)
conv2D4 = Conv2D(16, (2,2), activation='relu')(conv2D3)
conv2D5 = Conv2D(16, (2,2), activation='relu')(conv2D4)
drop2 = Dropout(0.2)(conv2D5)
conv2D6 = Conv2D(16, (2,2), activation='relu')(drop2)
conv2D7 = Conv2D(8, (5,5), activation='relu')(conv2D6)
gap = GlobalAveragePooling2D()(conv2D7)
dense1 = Dense(8, activation='relu')(gap)
drop3 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu')(drop3)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1) 

model.summary() # 30,970


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




#4. 평가, 예측
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




# GPU
# 걸린시간 :  124.68 초
# loss :  0.04402141645550728
# acc :  0.9904999732971191
# acc_score : 0.9905


# Model 함수
# 걸린시간 :  445.01 초
# loss :  0.2646692395210266
# acc :  0.9287999868392944
# acc_score : 0.9288


