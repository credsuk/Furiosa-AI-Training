import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

a = np.array([[1,2,3,4,5,6,7,8,9,10], # 이건 삼성주가
              [9,8,7,6,5,4,3,2,1,0], # 이건 온도
              ]).T # (2, 10) > (10, 2) 변경

print(a.shape) # (10 ,2)

"""
size = 5
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

print(bbb.shape) # (6, 5, 2)
# 슬라이싱(Slicing) 연산자
# x = bbb[:, :-1, :] # 모든 행에, 모든 것(:) / 배치사이즈, 타임스텝스, 특성
x = bbb[:, :-1]
# y = bbb[:, -1, 1] # 
y = bbb[:, -1, -1]

print(x.shape, y.shape)

"""
def split_x(dataset, timestep):
    x = []
    y = []
    timestep = timestep + 1
    for i in range(len(dataset) - timestep): 
        subset = dataset[i : (i + timestep)]
        x.append(subset[ : -1])
        y.append(subset[-1 : ])
    return np.array(x), np.array(y)

timestep = 4
x, y = split_x(a, timestep)

# print(y)  #[[[5 5]], [[6 4]], [[7 3]], [[8 2]], [[9 1]]]
# print(y.shape) # (5, 1, 2)

y = y[:, 0, 1] # Numpy 플러그인 / 행 : 첫번째 열 : 두번째 열


print(x.shape, y.shape) # (5, 4, 2) (5,)



#2 모델구축
model = Sequential()
model.add(LSTM(15, input_shape=(4,2)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련
learning_rate = 0.02
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
        batch_size=12,
        validation_split=0.01,
        callbacks=[es, rlr],
        )


#4. 평가, 예측

result = model.evaluate(x, y)
print("loss : ", result)

x_predict = np.array([[6, 4], [7, 3], [8, 2], [9, 1]]).reshape(-1, 4, 2)

y_predict = model.predict( x_predict )
print("y_predict 결과 : ", y_predict)


