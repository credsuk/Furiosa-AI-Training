import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.models import Sequential # 순차적인 형태인 모델로
from tensorflow.keras.layers import Dense # 밀집한 레이어 층
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # loss는 오차라는 뜻 이걸 mse방식으로 하겠다 optimizer는 우선 
model.fit(x, y, epochs=100) # fit 훈련시키다는 명령. 대부분 훈련이라는뜻 / x, y 훈련시키는 데이터 / epochs 훈련 횟수

#4. 평가 예측
result = model.predict(np.array([4])) # 4의 예측값을 추출하는 방법
print("4의 예측값 : ", result)


# Epoch 1000/1000
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - loss: 0.0021
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
# 4의 예측값 :  [[4.006394]]