"""
keras14_kaggle_bike1.py 복사
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time # 시간에 관한 플러그인



#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.shape, test_csv.shape, submission_csv.shape) # (10886, 11) (6493, 8) (6493, 2)

x = train_csv.drop(["casual", "registered", "count"], axis=1)
y = train_csv['count']
# print(x.shape, y.shape) # (10886, 8) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5437089)


#2. 모델구축
model = Sequential()
model.add(Dense(3, activation='relu', input_dim=8))
model.add(Dense(9, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time() # 현재 시간을 반환
model.fit(x_train, y_train, epochs=1, batch_size=32)
end_time = time.time() 


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse :", rmse)

print("걸린시간 : ", round(end_time - start_time, 2), "초")
# round() 반올림 해주는 함수, 뒤에가 소수점 자리수의 표시

"""
y_summit = model.predict(test_csv)
print(np.min(y_summit), np.max(y_summit))

submission_csv['count'] = y_summit
submission_csv.to_csv(path + "submit/submit_0904_1653.csv")

"""