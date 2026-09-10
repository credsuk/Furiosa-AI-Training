
"""
keras19_overfit1_califonia.py 복사 해옴

"""
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


# 스케일링을 한다
"""
MinMaxScaler

원값 - Min
--------------
Max - Min
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # 카이킥 런에서 # 핏은 실행하다 라는 뜻 아직 실행한거 아님 준비 까지
x = scaler.transform(x) # transform은 이걸 실행하는거
# print(x);
print(np.min(x), np.max(x)) # 0.0 1.0000000000000002 < 1이 넘어가는건 파이썬자체의 오류이다


# a = 0.1
# b = 0.2
# print(a+b) # 0.30000000000000004
# 컴퓨터인지 알아내는 방법...? 부동소수점 계산이라 이런 오차가 있다

print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)


#2. 모델구성
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(8,)))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2)
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("========================================================")
print("훈련에 걸린시간 : ", round(end_time - start_time, 2), "초")
print("loss : ", loss)

hist_loss = hist.history['loss']
hist_val_loss = hist.history['val_loss']


import matplotlib.pyplot as plt # 그림을 그려주는 플러그인

plt.rcParams["font.family"] = "Malgun Gothic" # 한글 깨짐때문에 한글폰트를 지정한다
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

plt.figure(figsize=(9,6)) # 가로/세로 그래프(그림)의 전체 크기를 설정
plt.plot(hist_loss[3:], c='red', label='loss') # y값만 넣으면 시간순으로 선을 그려줌 / c는 컬러 label은 명칭
plt.plot(hist_val_loss[3:], c='blue', label='val_loss')
# 초기에 loss가 너무 크니깐 3부터 시작하게 변경한다 (그래프의 모양이 이상해짐)

plt.legend(loc='upper right') # loc=위치(로케이션), 라벨의 위치 지정 위쪽에 오른쪽
plt.title('캘리포니아 loss') # 제목

plt.xlabel('epochs') # x축의 이름
plt.ylabel('loss') # y축의 이름

plt.grid() # 모눈종이 형태로 보여준다
# plt.show() # 표시

# 훈련에 걸린시간 :  198.05 초
# loss :  0.326321005821228