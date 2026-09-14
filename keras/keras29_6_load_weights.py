
"""
keras29_5_save_weights.py 복사 해옴

"""
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)


# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train) # 여기선 비율이 정해지고
x_train = scaler.transform(x_train) # 여기서 실제 변환을 한다
x_test = scaler.transform(x_test) # 같은 비율로 테스트 데이터를 변환한다


#2. 모델구성
model = Sequential()
model.add(Dense(14, activation='relu', input_shape=(8,)))
model.add(Dense(21, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))


model_path = "./_save/keras29/"

# 모델 저장
# model = model.save(model_path + "keras29_1_save_model.keras")

# Weights 저장 / 확장자를 h5으로 해야함
# model.save_weights(model_path + "keras29_5_save_weights1.weights.h5")

# 모델 불러오기
# model = load_model(model_path + "keras29_1_save_model.keras")

# Weights 불러오기
# Weights는 웨이트값만 저장하기 때문에 모델이 따로 있어야 한다
# 그래서 해당 저장 및 로드는 의미가 없다
# model.load_weights(model_path + "keras29_5_save_weights1.weights.h5")
model.load_weights(model_path + "keras29_5_save_weights2.weights.h5")



model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=50,
#     restore_best_weights=True
# )

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=500, batch_size=87, validation_split=0.2, callbacks=[es],)
# end_time = time.time()


# 훈련이 끝난 데이터 저장
# model.save(model_path + "keras29_3_save_model.keras")

# Weights 저장
# model.save_weights(model_path + "keras29_5_save_weights2.weights.h5")

# Weights 불러오기
# model.load_weights(model_path + "keras29_5_save_weights2.weights.h5")



#4. 평가 예측
print("================= keras29_5_save_weights ================")

loss = model.evaluate(x_test, y_test)
# print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss : ", loss)

# StandardScaler
# 훈련에 걸린시간 :  189.64 초
# loss :  0.2890736758708954

# 훈련에 걸린시간 : 46.99 초
# loss :  0.28104743361473083

# 훈련에 걸린시간 : 19.6 초
# loss :  0.30067992210388184

