import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

#1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96,106)) # 101 ~ 106까지 찾자.
size = 6

# 데이터를 reshape한 후 , split_x 함수로 시계열 데이터로 변환
# (N, 10, 1) -> (N, 5, 2)
a = a.reshape(-1, 2)
print(a.shape) # (50, 2)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

print(bbb.shape) # (45, 6, 2)

x = bbb[:, :-1, :]
y = bbb[:, -1:, 1]
print(x.shape, y.shape) # (45, 5, 2) (45, 1)



#2. 모둘 구성
model = Sequential()
model.add(LSTM(30, input_shape=(5,2)))
model.add(Dense(32, activation='relu'))
model.add(Dense(322, activation='relu'))
model.add(Dense(52, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()



#3 컴파일, 훈련
learning_rate = 0.001

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
        # batch_size=12,
        validation_split=0.01,
        callbacks=[es, rlr],
        )



#4. 평가, 예측
result = model.evaluate(x, y)
print("loss : ", result)


x_predict = np.array(range(96,106)) # 101 ~ 106까지 찾자.
print(x_predict.shape) # (6,)
# print(x_predict)

x_predict = x_predict.reshape(-1, 5, 2)
# print(x_predict)

y_predict = model.predict( x_predict )
print("y_predict 결과 : ", y_predict)



# 목표치 loss 0.1 이하
# 결과는
# [101, 102, 103, 104, 105, 106] 의 근사치가 나오면 됨
# loss :  0.043142132461071014
#  [[109.08735 ]
#  [109.58216 ]
#  [109.97884 ]
#  [110.420525]
#  [110.84472 ]
#  [111.27856 ]]




