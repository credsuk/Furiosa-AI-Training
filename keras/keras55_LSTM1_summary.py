# keras54_RNN2_summary 카피


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# 데이터가 얼마 없으니 수동으로
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ]) 

y = np.array([4,5,6,7,8,9,10]) # (7,)
# print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1) # 3차원으로 변경
# print(x.shape) # (7, 3, 1)


#2. 모델구성
model = Sequential()
model.add(LSTM(1, input_shape=(3, 1))) 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()
"""
Cell State
Cell State는 시간에 따라 변하지 않고 정보를 유지하는 경로

forget gate
현재 시점에서 이전 cell state에 저장된 정보 중 어떤 부분을 잊어야 할지 결정

input gate
input gate는 새로운 정보를 cell state에 얼마나 추가할지를 결정하는 중요한 역할

cell update  
cell update는 LSTM에서 forget gate와 input gate를 사용해 cell state를 업데이트

output gate
Output gate는 LSTM에서 현재 시점의 hidden state를 결정하는 데 중요한 역할


(gate + state) *  (( units 개수 * units 개수 ) + ( feature 수 + units 개수 ) + (bias * unit 개수))

"""
exit()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.02


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    factor=0.1 # 나누는 기준 기본값은 0.1
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
        batch_size=32,
        validation_split=0.1,
        callbacks=[es, rlr],
        )


#4. 평가, 예측
result = model.evaluate(x, y)
print("loss : ", result)

x_predict = np.array([8,9,10]).reshape(1, 3, 1)
y_predict = model.predict(x_predict)
print("[8,9,10]의 결과 : ", y_predict)






# 0.17280562222003937
# [8,9,10]의 결과 :  [[10.390333]]