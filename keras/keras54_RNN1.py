import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

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
# (7, 3)

y = np.array([4,5,6,7,8,9,10]) # (7,)
# print(x.shape, y.shape) # (7, 3) (7,)

# RNN은 3차원 텐서의 데이터
# 피처 데이터

x = x.reshape(x.shape[0], x.shape[1], 1) # 3차원으로 변경
# print(x.shape) # (7, 3, 1)


#2. 모델구성
model = Sequential()
# units 통상 output이라고 했던놈
# 어느 모델이든 제일 앞에있는 행 빼고 shape다 / 행무시, 열우선
# 100, 100, 3 데이터도 가능 이미지형 데이터
# model.add(SimpleRNN(units=10, input_shape=(3, 1))) 
model.add(SimpleRNN(16, input_shape=(3, 1))) 

# 3차원으로 들어가서 1차원, 2차원으로 나온다 그래서 바로 Dense와 연결가능
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


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