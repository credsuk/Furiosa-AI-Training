
# sklearn에서 데이터를 못불러 올때 ssl 오류로 못불러 올 수 있음 
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing # 캘리포니아 집값
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
# 연습용 캘리포니아 집값 데이터로 모델 구성하고 훈련시키고 평가하기
datasets = fetch_california_housing() # 리턴값을 받아서
x = datasets.data
y = datasets.target

print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)


#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=8))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("========================================================")
print("loss : ", loss)


# loss :  0.6139959096908569