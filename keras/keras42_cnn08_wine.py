# keras23_softmax2_wine.py 복사

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48]))

y = pd.get_dummies(y, dtype=int)
print(x.shape, y.shape) # (178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=4312,
    stratify=y,
)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape) # (142, 13) (36, 13)


x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(15, (2,1), padding='same', input_shape=(13,1,1), activation='relu'))
model.add(Conv2D(10, (2,1), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(24, activation='relu'))
model.add(Dense(3, activation='softmax'))



#3. 컴파일, 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'],
)

es = EarlyStopping(
    monitor='var_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
)
end_time = time.time()


#4. 평가 예측
print("================= wine =================")
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print("acc_score : ", acc_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")




# acc = 0.95 이상 합격


# MinMaxScaler
# loss : 0.050119463354349136
# acc : 0.9722222089767456
# acc_score :  0.9722222222222222
# 걸린시간 :  12.32 초

# loss : 0.47242915630340576
# acc : 0.8055555820465088
# acc_score :  0.8055555555555556
# 걸린시간 :  15.36 초