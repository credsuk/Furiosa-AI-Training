# keras36_cnn3_mnist 복사

import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))  # 1.0 -1.0
# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

# 4차원 데이터로 변환을 위해서 reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# print(x_train.shape, x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)




#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
# 위에 코드로 커널링 후에 (26, 26, 64) 로 변경됨 (28-3) + 1, (28-3) + 1, 노드의 수

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu')) # (23, 23, 32)
model.add(Conv2D(16, (2,2), activation='relu')) # (22, 22, 6)
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu')) # (21, 21, 16)
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu')) # (20, 20, 16)
model.add(Conv2D(8, (5,5), activation='relu')) # (16, 16, 8) # 추가

model.add(GlobalAveragePooling2D())
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax')) # (10, )

model.summary()


#3. 컴파일, 훈련
# sparse_을 넣으면 y값이 OneHot을 할 필요가 없다 알아서 해준다
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
    epochs=10000,
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

# exit()
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

# print(y_predict)
# print(y_test)

print(y_predict.shape, y_test.shape) # (10,) (10000,)
# y_test = np.argmax(y_test, axis=1)

print(y_predict.shape, y_test.shape) # (10,) (10000,)
acc_score = accuracy_score(y_test, y_predict)
print("acc_score :", acc_score)



# 걸린시간 :  509.7 초
# loss :  0.3589233160018921
# acc :  0.9016000032424927
# acc_score : 0.9016




