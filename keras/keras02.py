from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000)

#4. 평가 예측
loss = model.evaluate(x, y) # 모델을 평가하는거
print("loss : ", loss)

result = model.predict(np.array([1,2,3,4,5,6,7]))
print("7의 예측값 : ", result)


# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 3.9790e-13
# loss :  3.979039320256561e-13
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
# 7의 예측값 :  [[1.000001 ]
#  [2.0000005]
#  [3.0000002]
#  [3.9999998]
#  [4.9999995]
#  [5.999999 ]
#  [6.9999986]]