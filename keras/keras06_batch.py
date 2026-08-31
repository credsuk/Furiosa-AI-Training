from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# 훈련을 시킬 때 데이터를 나누는걸 Batch라고 한다 batch_size=3은 3개씩 잘라서 훈련을 시키겠다라는 뜻이다.
# batch_size를 설정하지 않으면 기본값은 32이다. 즉, 32개씩 잘라서 훈련을 시키겠다라는 뜻이다.
# 이것도 하이퍼 파라미터 튜닝이다.
model.fit(x, y, epochs=1000, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

