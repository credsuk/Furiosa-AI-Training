import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

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

x = x.reshape(x.shape[0], x.shape[1], 1) # 3차원으로 변경
# print(x.shape) # (7, 3, 1)


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(16, input_shape=(3, 1))) 
# Bidirectional자체는 모델이 아님 이건 그냥 양방향으로 돌리겠다 라는 뜻
# 래핑 > 모델을 레핑하겠다 그래서 아래와 같이 사용해야함
model.add(Bidirectional(SimpleRNN(16), input_shape=(3, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()


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

