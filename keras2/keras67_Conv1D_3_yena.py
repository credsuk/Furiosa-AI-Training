# keras66_jena_CNN 복사
# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# 메모리 터질때 이거하면 메모리를 모아서 안터질 수도 있다
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # 메모리 모으기


import numpy as np
import pandas as pd
import time, datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#1. 데이터
path = "./_data/kaggle_jena/"

datasets = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0,) 
# print(datasets.shape) # (420551, 14)

y_cor = datasets[-144:]['T (degC)'] # 예측치 정답 데이터
# print(y_cor.shape) # (144,)


### 훈련할 데이터 자르기 ###
x_data = datasets[:-288].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
y_data = datasets[144: -144]['T (degC)'].to_numpy(dtype=np.float32)

# print(x_data.shape, y_data.shape) # (420263, 13) (420263,)


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

# print(x_data.shape, y_data.shape) # (420120, 144, 13) (420120, 144)
print("자르는시간 : ", round(end_time - start_time, 2), "초") # 자르는시간 :  24.0 초


# summit용 x 데이터
x_predict = datasets[-288:-144].drop(['wd (deg)'], axis=1)
# print(type(x_predict)) # <class 'pandas.core.frame.DataFrame'>
x_predict = x_predict.to_numpy()
# print(x_predict.shape) # (144, 13)
x_predict = x_predict.reshape(1, 144, 13)


# 훈련, 테스트 데이터 분리 train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=523, shuffle=False)
# print(x_train.shape, y_train.shape) # (315090, 144, 13) (315090, 144)



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
model.add(Conv1D(36, (4), input_shape=(144, 13)))
model.add(Dropout(0.3))
model.add(Conv1D(48, (4)))
model.add(Dropout(0.2))
model.add(Conv1D(24, (2)))
model.add(GlobalAveragePooling1D())
model.add(Dense(108, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(28, activation='relu'))
model.add(Dense(144, activation='relu'))

model.summary()
# exit()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'],)


mcp = ModelCheckpoint(
    monitor='val_loss', # 모니터되는 데이터
    mode='auto', # 자동으로
    save_best_only=True, # 최고의 값만 저장해라
    # save_weights_only=True,
    filepath="./_save/keras67/keras67_Conv1D_3_yena_{epoch:04d}-{val_loss:.4f}.keras", # 파일 경로
    verbose=1, # 보여줄지 안보여줄지 확인하는 항목
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
        epochs=100000,
        batch_size=500,
        validation_split=0.2,
        callbacks=[es, rlr, mcp],
        )
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : # def로 함수를 정의하고 이름을 뒤에 쓴다 
    return np.sqrt(mean_squared_error(y_test, y_predict)) # sqrt는 루트를 거는 함수
rmse = RMSE(y_test, y_predict)

print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss :", loss[0])
print("acc :", round(loss[1], 4))
print("RMSE : ", rmse)



# 훈련에 걸린시간 : 264.27 초
# loss : 76.50410461425781
# acc : 0.0383
# RMSE :  8.74613256816419

# 훈련에 걸린시간 : 704.76 초
# loss : 23.988069534301758
# acc : 0.0289
# RMSE :  4.897690224723641

