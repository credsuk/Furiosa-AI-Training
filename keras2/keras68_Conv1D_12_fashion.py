# keras52_opmizer12_fashion 복사

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import time



#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)

# MaxAbsScaler 데이터 스케일링
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# x 데이터 4차원 데이터로 변경
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

# OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))
# print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(32, (2,), input_shape=(28, 28)))
model.add(Dropout(0.2))
model.add(Conv1D(64, (3,), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv1D(72, (2,), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(41, (3,), activation='relu'))

model.add(Flatten())

model.add(Dense(44, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) # (10, )

# model.summary()


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.005

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc']) 

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
import z_evaluate
z_evaluate.evaluate(model, x_test, y_test, start_time, end_time, name="12_fashion")
# loss = model.evaluate(x_test, y_test, verbose=1)
# print("걸린시간 : ", round(end_time - start_time, 2), "초")
# print("================= model.evaluate ===========================")
# print("loss : ", loss[0])
# print("acc : ", loss[1])

# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
# y_test = np.argmax(y_test, axis=1)
# acc_score = accuracy_score(y_test, y_predict)
# print("acc_score :", acc_score)




# 0.92 acc 기준

# 걸린시간 :  222.52 초
# loss :  0.2419130653142929
# acc :  0.91839998960495
# acc_score : 0.9184


# learning_rate = 0.005
# loss :  0.287946879863739
# acc :  0.8992000222206116
# acc_score : 0.8992

# =================12_fashion ================
# 훈련 시간 : 80.15 초
# loss : 0.3494662940502167
# acc : 0.8738
# r2 : 0.8031666112329703
# mse : 0.017715004989030912
