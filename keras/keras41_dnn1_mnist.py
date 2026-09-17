# keras39_MaxPooling1_mnist 복사
# 
import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder # 분류이기 때문에 OneHot를 해야함
from sklearn.metrics import accuracy_score



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 스케일링
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# 4차원 데이터로 변환을 위해서 reshape
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
# 여기서는 2차원으로 변경할꺼댜
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)

print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)


# Y 데이터의 원핫
ohe = OneHotEncoder(sparse_output=False) # 뭔가 혼동스러운 행열(혼동 행열)을 방지하기 위해서 sparse_output 사용함
y_train = y_train.reshape(-1, 1) # 수치를 알면 60000, 1 이렇게 해도 되지만 모르면 전체라는 의미에서 -1, 1 이렇게 하면된다
y_test = y_test.reshape(-1, 1) # 사실상 -1, 1 이게 디폴트

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)



#2. 모델구성
model = Sequential()
model.add(Dense(64, input_shape=(28*28,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(84, activation='relu'))
model.add(Dense(168, activation='relu'))
model.add(Dense(248, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) 
# model.add(Dropout(0.2))

# model.add(Conv2D(32, (2,2), activation='relu')) # (23, 23, 32)

# model.add(Conv2D(16, (2,2), activation='relu')) # (22, 22, 6)
# model.add(Dropout(0.2))


# model.add(Conv2D(16, (2,2), activation='relu')) # (21, 21, 16)
# model.add(Dropout(0.2))

# model.add(Conv2D(16, (2,2), activation='relu')) # (20, 20, 16)

# # model.add(Flatten())
# model.add(GlobalAveragePooling2D())

# model.add(Dense(units=8, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax')) # (10, )

model.summary()



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


# Flatten
# 걸린시간 :  451.76 초
# loss :  0.04781254008412361
# acc :  0.9860000014305115
# acc_score : 0.986


# GlobalAveragePooling2D
# 걸린시간 :  161.18 초
# loss :  0.051816169172525406
# acc :  0.9851999878883362
# acc_score : 0.9852

# Dense
# 걸린시간 :  89.99 초
# loss :  0.09426125884056091
# acc :  0.9745000004768372
# acc_score : 0.9745

