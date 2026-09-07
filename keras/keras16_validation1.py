"""
keras15_verbose.py 카피
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


x_train = np.array([1,2,3,4,5,6]) 
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])


#2. 모델
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 4, 
          verbose = 1,
          validation_data = (x_val, y_val) # 훈련시 valid데이터로 predict를 해서 값을 보여준다
          )
# loss값은 과적합된 데이터 이다
# 조금 더 신뢰할 데이터가 필요하기 때문에 valid데이터를 생성하여 넣는다
# 해당 값이 안정적으로 떨어진다면 정상적이다
# 우리가 좀 더 신뢰하고 확인해야할 데이터는 val_loss 데이터다
# 통상적으로 loss가 더 좋고, val_loss가 더 안좋다


#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)


