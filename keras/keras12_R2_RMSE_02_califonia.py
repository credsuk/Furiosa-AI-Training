from sklearn.datasets import fetch_california_housing # 캘리포니아 집값
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=43)

print(x_train.shape, y_train.shape) # (15480, 8) (15480,)


#2. 모델구성
model = Sequential()
model.add(Dense(2, input_dim=8))
model.add(Dense(10))
model.add(Dense(11))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=24)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

# R2 기준 0.55

# loss :  0.5950935482978821
# r2 :  0.5517865195378517
