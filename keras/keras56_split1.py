import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# 시계열 데이터 짜르기

a = np.array(range(1, 11))
size = 5 # timestep 사이즈

print(a.shape) # (10,)

# 이 함수만 이해하면 시계을 데이터는 짜를 수 있다
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)



bbb = split_x(a, size)
# print(bbb.shape) # (6, 5)
# print(bbb)
# y 데이터는 끝에만 짜르면 된다


x = bbb[:, :-1]
y = bbb[:, -1:]

print(x)
print(y)

print(x.shape, y.shape) # (5, 5) (1, 5)

exit()

def timestemp_x(dataset, timestep):
    x = []
    y = []
    timestep
    for i in range(len(dataset) - timestep + 1): 
        subset = dataset[i : ((i + timestep))]
        x.append(subset[ : -1])
        y.append(subset[-1 : ])
    return np.array(x), np.array(y)

timestep = 5
x, y = timestemp_x(a, timestep)

print(x.shape, y.shape) # (6, 4) (6, 1)
# print(y)


#2 모델구축
model = Sequential()
model.add(LSTM(10, input_shape=(4,1)))
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

y_predict = model.predict(np.array([[7,8,9,10]]))
print("y_predict 결과 7,8,9,10 예측값 : ", y_predict)


