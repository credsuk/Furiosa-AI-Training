# keras28_Scaler09_fetch_covtype.py 복사

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


import numpy as np
import pandas as pd
import time, datetime


#1. 데이터
datasets = fetch_covtype()
# print(datasets.DESCR) # 용량이 커서 이거 하면 안됨
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

y = pd.get_dummies(y, dtype=int)
print(x.shape, y.shape) # (581012, 54) (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    random_state=513,
    stratify=y,
)


scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
model = Sequential()
model.add(Dense(73, input_dim=54, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.24))
model.add(Dense(72, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(45, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(21, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['acc'],
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=10000,
    batch_size=6931,
    validation_split=0.2,
    callbacks=[es, ],
)
end_time = time.time()


#4. 평가 예측
print("================= keras31_MCP_save_09_fatch_covtype =================")

result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print("acc_score : ", acc_score)
print("걸린시간 :", round(end_time - start_time, 2), "초")


## acc = 0.93 이상


# MaxAbsScaler
# loss : 0.24471095204353333
# acc : 0.9050966501235962
# acc_score :  0.9050966245103371
# 걸린시간 : 353.14 초


# loss : 0.3968348205089569
# acc : 0.8359827399253845
# acc_score :  0.8359827335752101
# 걸린시간 : 63.13 초

# loss : 0.4499477446079254
# acc : 0.8139452934265137
# acc_score :  0.8139453229881655
# 걸린시간 : 85.91 초