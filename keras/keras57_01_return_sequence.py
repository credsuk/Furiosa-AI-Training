import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
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


#2. 모델구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3,1), return_sequences=True,))
model.add(LSTM(15, return_sequences=True))
model.add(GRU(36, return_sequences=True))
model.add(GRU(24, return_sequences=True))
model.add(GRU(10, return_sequences=True))
model.add(LSTM(6))
# model.add(Dense(32))
# model.add(Dense(54))
model.add(Dense(1))

model.summary()


#3. 컴파일 훈련
learning_rate = 0.002
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=60,
    verbose=1,
    factor=0.2 # 나누는 기준 기본값은 0.1
)

es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=100,
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
print("loss : ", result)

y_predict = model.predict(x_predict)
print("y_predict 결과 : ", y_predict)







# loss :  38.49064254760742
# y_predict 결과 :  [[49.99999]]

