"""
keras12_R2_RMSE_03_diabetes.py 복사
"""
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 


#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=34)

# print(x_train.shape, y_train.shape) # (331, 10) (331,)

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=10))
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=700, batch_size=27, validation_split=0.2)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

# R2 기준 0.62
# loss :  2974.061767578125
# r2 : 0.42029513858750356

"""
# validation_split 추가 후 모니터링 튜닝결과

"""
