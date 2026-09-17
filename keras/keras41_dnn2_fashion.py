# keras39_MaxPooling2_fashion 복사

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
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
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# 2차원 데이터로 변경
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

# OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))
# print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)


#2. 모델 구성
model = Sequential()
model.add(Dense(82, input_shape=(28*28,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(168, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(316, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(214, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(10, activation='softmax')) # (10, )

# model.summary()


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


# Flatten
# 걸린시간 :  222.52 초
# loss :  0.2419130653142929
# acc :  0.91839998960495
# acc_score : 0.9184



# GlobalAveragePooling2D
# 걸린시간 :  228.09 초
# loss :  0.24673676490783691
# acc :  0.9147999882698059
# acc_score : 0.9148


# Dense
# 걸린시간 :  55.19 초
# loss :  0.3330053687095642
# acc :  0.8805000185966492
# acc_score : 0.8805