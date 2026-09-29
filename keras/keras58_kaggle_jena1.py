# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data


# 메모리 터질때 이거하면 메모리를 모아서 안터질 수도 있다
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # 메모리 모으기

import numpy as np
import pandas as pd
import time, datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#1. 데이터
path = "./_data/kaggle_jena/"

# 2016.12.31 00시 ~ 2017. 01. 01 00시 데이터 맞추기
# 10분마다 있기 때문에 144개가 타임 스탭프
# 원본데이터 12월 31일 데이터는 삭제 해야됨
# wd(deg)값을 y 하자

"""
# 너무 어렵게 생각 했나봄 (기억용)

read_csv = pd.read_csv(path + "jena_climate_2009_2016.csv", ) 
print(read_csv.shape) # (420551, 14)

print(read_csv.columns)

# 'Date Time' 열을 날짜/시간 타입으로 변환
read_csv['Date Time'] = pd.to_datetime(read_csv['Date Time'], format='%d.%m.%Y %H:%M:%S')

# 기준값(2016-12-31 00:00:00) 이하인 데이터만 남기기 (초과하는 모든 행 제거)
current_date = pd.to_datetime('2016-12-31 00:00:00')
read_csv = read_csv[read_csv['Date Time'] <= current_date]

# Date Time 컬럼 삭제
read_csv = read_csv.drop(['Date Time'], axis=1) 
print(read_csv.shape) # (420407, 14)


x_data = read_csv.drop(['wd (deg)'], axis=1)
y_data = read_csv['wd (deg)'] # 예측치 정답 데이터

print(x_data.shape, y_data.shape) # (420407, 13) (420407,)
"""


# 144 = 1일

datasets = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0,) 
print(datasets.shape) # (420551, 14)

y_cor = datasets[-144:]['wd (deg)'] # 예측치 정답 데이터
print(y_cor.shape) # (144,)


### 훈련할 데이터 자르기 ###
x_data = datasets[:-288].drop(['wd (deg)'], axis=1).to_numpy(dtype=np.float32)
y_data = datasets[144: -144]['wd (deg)'].to_numpy(dtype=np.float32)

print(x_data.shape, y_data.shape) # (420263, 13) (420263,)


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

size_x = 144
size_y = 144

start_time = time.time()
x_data = split_x(x_data, size_x)
y_data = split_x(y_data, size_y)
end_time = time.time()

print(x_data.shape, y_data.shape) # (420120, 144, 13) (420120, 144)
print("자르는시간 : ", round(end_time - start_time, 2), "초") # 자르는시간 :  24.0 초


# summit용 x 데이터
x_predict = datasets[-288:-144].drop(['wd (deg)'], axis=1)
print(type(x_predict)) # <class 'pandas.core.frame.DataFrame'>
x_predict = x_predict.to_numpy()
print(x_predict.shape) # (144, 13)
x_predict = x_predict.reshape(1, 144, 13)


# 훈련, 테스트 데이터 분리 train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=5123, shuffle=False)
print(x_train.shape, y_train.shape) # (315090, 144, 13) (315090, 144)


# Scaler
print(x_train.shape, x_test.shape) # (315090, 144, 13) (105030, 144, 13)
x_train_scaler = x_train.reshape(-1, x_train.shape[-1])
x_test_scaler = x_test.reshape(-1, x_test.shape[-1])

scaler = MinMaxScaler()
scaler.fit(x_train_scaler)
x_train_scaler = scaler.transform(x_train_scaler)
x_test_scaler = scaler.transform(x_test_scaler)
print(x_train_scaler.shape, x_test_scaler.shape) # (45372960, 13) (15124320, 13)

x_train = x_train_scaler.reshape(x_train.shape)
x_test = x_test_scaler.reshape(x_test.shape)
print(x_train.shape, x_test.shape) # (315090, 144, 13) (105030, 144, 13)


#2. 모델 구축
model = Sequential()
model.add(LSTM(20, input_shape=(144, 13)))
model.add(Dense(144))



#3. 컴파일, 훈련
learning_rate = 0.006
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=60,
    verbose=1,
    factor=0.1 # 나누는 기준 기본값은 0.1
)

es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=20,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train, 
        epochs=2000,
        batch_size=500,
        validation_split=0.2,
        callbacks=[es, rlr],
        )
end_time = time.time()


# 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = model.predict(x_predict)
# r2 = r2_score(y_test, y_predict)


print("훈련 걸린시간 : ", round(end_time - start_time, 2), "초")
print("loss : ", result)
print("y_predict : ", y_predict)
# print("r2 :", r2)


# 자르는시간 :  1.56 초


# 훈련 걸린시간 :  323.7 초
# loss :  6762.5537109375
