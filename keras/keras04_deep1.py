from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성 
# 레이어를 층별로 딥러닝하게 만든다
# 이런 일련의 과정을 하이퍼 파라미터라고 한다 그걸 튜닝을 해서 하이퍼 파라미터 튜닝
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(10, input_dim=50))
model.add(Dense(5, input_dim=10))
model.add(Dense(5, input_dim=5))
model.add(Dense(1, input_dim=5))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

result = model.predict(np.array([1,2,3,4,5,6]))
print("6의 예측값 : ", result)



