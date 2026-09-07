
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

# 최소값을 구하는 방법
# 미리 만들어져 있는  클래스 임포트
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss', # 기준을 잡는다 / loss가능 val_loss가 제일 믿음이 가니 이걸로
    mode = 'min', # 어떤 알고리즘을 찾을것인가? 잘 모르겠을때는 auto 액큐러시지표는 max 우리는 loss를 잡는거니깐 min
    patience = 50 , # 인내심 / 얼마나 참을 것인가 / 우린 20번 참는다
    restore_best_weights = True, # 최적의 웨이트 지점에서 값을 리턴한다
)
# 원칙적으로는 restore_best_weights 값이 좋아야 하는데.. 실무를 뛰다보면 꼭 그렇지 않다 비교 해보고 좋은걸 써라

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=10000, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es], # EarlyStopping을 적용해서 반영해라 / 리스트 형태라 더 적용할 수 있다(더있다.. 아직 모름)
                 )
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("========================================================")
print("훈련에 걸린시간 : ", round(end_time - start_time, 2), "초")
print("loss : ", loss)



import matplotlib.pyplot as plt # 그림을 그려주는 플러그인

plt.rcParams["font.family"] = "Malgun Gothic" # 한글 깨짐때문에 한글폰트를 지정한다
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

plt.figure(figsize=(9,6)) # 가로/세로 그래프(그림)의 전체 크기를 설정
plt.plot(hist.history["loss"][3:], c='red', label='loss') # y값만 넣으면 시간순으로 선을 그려줌 / c는 컬러 label은 명칭
plt.plot(hist.history["val_loss"][3:], c='blue', label='val_loss')
# 초기에 loss가 너무 크니깐 3부터 시작하게 변경한다 (그래프의 모양이 이상해짐)

plt.legend(loc='upper right') # loc=위치(로케이션), 라벨의 위치 지정 위쪽에 오른쪽
plt.title('캘리포니아 loss') # 제목

plt.xlabel('epochs') # x축의 이름
plt.ylabel('loss') # y축의 이름

plt.grid() # 모눈종이 형태로 보여준다
plt.show() # 표시
