# keras28_Scaler10_digits.py 복사

from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import time, datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = load_digits()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (1797, 64) (1797,)

############# 원핫2. 판다스 pd.get_dummies #############
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,
    random_state= 421, 
    shuffle = True,
    stratify = y, 
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
# model = Sequential()
# model.add(Dense(82, input_dim=64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))


input1 = Input(shape=(64,)) 
dense1 = Dense(82, activation='relu')(input1)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(64, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
output1 = Dense(10, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1) 


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=800,
    restore_best_weights=True
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es, ],
          )
end_time = time.time()


#4. 평가, 예측
print("================= keras31_MCP_save_10_digits =================")
result = model.evaluate(x_test, y_test)
print('loss = ', result[0])
print('acc = ', round(result[1], 2))

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
y_test = np.argmax(y_test, axis=1)
# print(y_test)

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score = ', accuracy_score)
print('걸린시간 = ', round(end_time - start_time, 2), "초")


# MaxAbsScaler
# loss =  0.086285300552845
# acc =  0.97
# acc_score =  0.9703703703703703
# 걸린시간 =  10.95 초


# loss =  0.0628291666507721
# acc =  0.98
# acc_score =  0.9777777777777777
# 걸린시간 =  10.13 초


# loss =  0.08172474801540375
# acc =  0.97
# acc_score =  0.9740740740740741
# 걸린시간 =  10.31 초


# loss =  0.0687260776758194
# acc =  0.98
# acc_score =  0.975925925925926
# 걸린시간 =  10.04 초

