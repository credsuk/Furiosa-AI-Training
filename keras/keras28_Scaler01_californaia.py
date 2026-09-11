
"""
keras27_Scaler01_californaia.py 복사 해옴

"""
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
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


print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# scaler.fit(x_train) # 여기선 비율이 정해지고
# x_train = scaler.transform(x_train) # 여기서 실제 변환을 한다
x_train = scaler.fit_transform(x_train) # 이렇게도 사용가능
x_test = scaler.transform(x_test) # 같은 비율로 테스트 데이터를 변환한다


print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test)) # -0.001071811361200048 2.313783684968923

# 기준에 x_train이고 같은 비율로 x_test를 해야 과적합되지 않는다
# x데이터 전체를 하면 x_test의 데이터의 데이터를 확정할 수 없다


#2. 모델구성
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(8,)))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[es],)
end_time = time.time()

#4. 평가 예측
print("================= keras28_Scaler01_californaia ================")

loss = model.evaluate(x_test, y_test)
print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss : ", loss)


hist_loss = hist.history['loss']
hist_val_loss = hist.history['val_loss']
# print("loss :", hist_loss)
# print("val_loss :", hist_val_loss)

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


# MinMaxScaler
# 훈련에 걸린시간 :  171.99 초
# loss :  0.3200035095214844


# StandardScaler
# 훈련에 걸린시간 :  189.64 초
# loss :  0.2890736758708954


# MaxAbsScaler
# 훈련에 걸린시간 : 326.86 초
# loss :  0.3972775936126709


# RobustScaler
# 훈련에 걸린시간 : 115.06 초
# loss :  0.3276841938495636