from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

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


#2. 모델구성
model = Sequential()
model.add(Dense(21, input_dim=13, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(7, activation='relu'))
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
    epochs=20000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
)
end_time = time.time()


#4. 평가 예측
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