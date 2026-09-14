
"""
keras29_6_load_weights.py 복사 해옴

"""
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import time

model_path = "./_save/keras30/"

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print("shape ::", x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)


# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train) # 여기선 비율이 정해지고
x_train = scaler.transform(x_train) # 여기서 실제 변환을 한다
x_test = scaler.transform(x_test) # 같은 비율로 테스트 데이터를 변환한다


# #2. 모델구성
model = Sequential()
model.add(Dense(14, activation='relu', input_shape=(8,)))
model.add(Dense(21, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))



# #3. 컴파일, 훈련
# # from tensorflow.keras.callbacks import ModelCheckpoint
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
    verbose=1, # 보여주는 내용
)

mcp = ModelCheckpoint(
    monitor='val_loss', # 모니터되는 데이터
    mode='auto', # 자동으로
    save_best_only=True, # 최고의 값만 저장해라
    filepath=model_path + "keras30_mcp1.keras", # 파일 경로
    verbose=1, # 보여줄지 안보여줄지 확인하는 항목
)

start_time = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=5000, 
    batch_size=87, 
    validation_split=0.2, 
    callbacks=[es, mcp], # mcp 추가
    verbose=1
)

# model = load_model(model_path + "keras30_mcp1.keras") # 가중치까지 모두 저장된 웨이트를 불러온다
end_time = time.time()



#4. 평가 예측
print("================= keras30_ModelCheckPoint1 ================")

loss = model.evaluate(x_test, y_test)
print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss : ", loss)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse :", rmse)

# StandardScaler
# 훈련에 걸린시간 :  189.64 초
# loss :  0.2890736758708954


# ================= keras30_ModelCheckPoint1 ================
# 훈련에 걸린시간 : 34.05 초
# loss :  0.29389050602912903
# r2 : 0.780414389907094
# rmse : 0.5421166236163238