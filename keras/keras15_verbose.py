"""
keras09_train_test1.py 카피
verbose > 뜻 : 말 수가 많은 
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


x_train = np.array([1,2,3,4,5,6,7]) 
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2. 모델
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4, 
          verbose=0 # 훈련되는 과정을 보여주지 말아라, 훈련은 된다 / 0은 가리고 1은 보이고
          )
# verbose = 0 : 침묵
# verbose = 1 : 디폴트, 다 보여줘
# verbose = 2 : 프로그레스바 생략 / 결과만 보임
# verbose = 3 : Epochs 값만 나오게 마지막에 로스값
# verbose = n : 그외 값들은 verbose = 3과 동일한 값만 나옴

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)


