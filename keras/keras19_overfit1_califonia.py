
"""
keras17_val1_califonia.py 복사 해옴

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

print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)


#2. 모델구성
model = Sequential()
model.add(Dense(4, activation='relu', input_dim=8))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("========================================================")
print("훈련에 걸린시간 : ", round(end_time - start_time, 2), "초")
print("loss : ", loss)

print("========================= hist ===============================")
print(hist)
print("========================= hist.history ===============================")
print(hist.history)
"""
{
    'loss': [65.28096771240234, 2.453090190887451, 1.1672496795654297, 0.8666049242019653, 0.7593405246734619, 0.7224177122116089, 0.6885923147201538, 0.6621494293212891, 0.6426905989646912, 0.6626430749893188], 
    'val_loss': [2.7848868370056152, 1.370016098022461, 0.9148955345153809, 0.8258680105209351, 0.6656901240348816, 0.6903218030929565, 0.6282101273536682, 0.8736544847488403, 0.564456582069397, 0.619134247303009]
}
"""
# 두개 이상은 List
# 키 : 밸류는 Dictionary
print("========================= loss ===============================")
hist_loss = hist.history['loss']
# print(hist_loss)
print("========================= val_loss ===============================")
hist_val_loss = hist.history['val_loss']
# print(hist_val_loss)
print("========================================================")


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
plt.show() # 표시
