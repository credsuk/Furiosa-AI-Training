# keras55_LSTM2_scale 복사

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60],
              ])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape, y.shape) # (13, 3, 1) (13,)


x_predict =  np.array([50,60,70]) # 80에 맞춘다
# print(x_predict.shape) # (3,)
x_predict = x_predict.reshape(-1, 3, 1)
# print(x_predict.shape) # (3, 1)


#2. 모델 구축
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(3, 1)))
model.add(Conv1D(20, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()
# exit()

#3. 컴파일, 훈련
learning_rate = 0.05
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=80,
    verbose=1,
    factor=0.1 # 나누는 기준 기본값은 0.1
)

es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=150,
    restore_best_weights=True,
)

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
model.fit(x, y, 
        epochs=50000,
        batch_size=12,
        validation_split=0.1,
        callbacks=[es, rlr],
        )


#4. 평가, 예측

result = model.evaluate(x, y)
y_predict = model.predict(x_predict)

print("loss : ", result)
print("y_predict 결과 : ", y_predict)



# loss :  7.206088776001707e-05
# y_predict 결과 :  [[80.020004]]

