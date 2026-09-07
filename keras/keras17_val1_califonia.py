"""
keras12_R2_RMSE_02_califonia.py 복사
"""
from sklearn.datasets import fetch_california_housing # 캘리포니아 집값
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=8))
model.add(Dense(7))
model.add(Dense(12))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.1)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("========================================================")
print("loss : ", loss)


# loss :  0.6139959096908569

"""
# validation_split 추가 후 모니터링 튜닝결과
loss :  0.7132998704910278
"""