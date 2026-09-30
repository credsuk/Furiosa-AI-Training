# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data


# 메모리 터질때 이거하면 메모리를 모아서 안터질 수도 있다
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # 메모리 모으기

import numpy as np
import pandas as pd
import time, datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, GRU, Dropout, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#1. 데이터
path = "./_data/kaggle_jena/"

# 2016.12.31 00시 ~ 2017. 01. 01 00시 데이터 맞추기
# 10분마다 있기 때문에 144개가 타임 스탭프
# 원본데이터 12월 31일 데이터는 삭제 해야됨
# T (degC)값을 y 하자

# 144 = 1일

datasets = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0,) 
print(datasets.shape) # (420551, 14)

y_cor = datasets[-144:]['T (degC)'] # 예측치 정답 데이터
print(y_cor.shape) # (144,)


### 훈련할 데이터 자르기 ###
x_data = datasets[:-288].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
y_data = datasets[144: -144]['T (degC)'].to_numpy(dtype=np.float32)

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
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=523, shuffle=False)
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
model.add(Bidirectional(LSTM(36, return_sequences=True), input_shape=(144, 13), ))
model.add(GRU(48))
model.add(Dense(286, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(62, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(144))

model.summary()
# exit()

#3. 컴파일, 훈련
learning_rate = 0.008
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

mcp = ModelCheckpoint(
    monitor='val_loss', # 모니터되는 데이터
    mode='auto', # 자동으로
    save_best_only=True, # 최고의 값만 저장해라
    # save_weights_only=True,
    filepath="./_save/keras58/kaggle_jena1_{epoch:04d}-{val_loss:.4f}.keras", # 파일 경로
    verbose=1, # 보여줄지 안보여줄지 확인하는 항목
    max_to_keep=5,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    factor=0.1 # 나누는 기준 기본값은 0.1
)

es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=100,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train, 
        epochs=5000,
        batch_size=700,
        validation_split=0.2,
        callbacks=[es, rlr, mcp],
        )
end_time = time.time()


# 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# print(y_test.shape, y_predict.shape) # (105030, 144) (1, 144)
def RMSE(y_test, y_predict) : # def로 함수를 정의하고 이름을 뒤에 쓴다 
    return np.sqrt(mean_squared_error(y_test, y_predict)) # sqrt는 루트를 거는 함수
rmse = RMSE(y_test, y_predict)



print("훈련 걸린시간 : ", round(end_time - start_time, 2), "초")
print("loss : ", result)
print("RMSE : ", rmse)


# 자르는시간 :  1.08 초
# 훈련 걸린시간 :  1158.68 초
# loss :  6.096022129058838
# RMSE :  2.468988141513063

