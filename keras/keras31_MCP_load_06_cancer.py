# keras28_Scaler06_cancer.py 불러옴

import numpy as np
import pandas as pd
import time
import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer # load_breast_cancer 유방암관련 데이터셋 / 분류형 데이터 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_breast_cancer()

x = datasets["data"]
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3,
    random_state=124, 
    stratify=y, # y데이터를 stratify해라 골고루 나오게 해라
)

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성


#3. 컴파일, 훈련
mcp_path = "./_save/keras31/cancer/"

start_time = time.time()
model = load_model(mcp_path + "k31_0914_1412_0026-0.1402.keras")
end_time = time.time()



#4. 평가, 예측
print("================= keras31_MCP_save_06_cancer =================")
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
# print(np.round(y_pred[:10]))

rmse = root_mean_squared_error(y_test, y_pred)

print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss :", loss[0])
print("acc :", round(loss[1], 4))
# print("rmse :", rmse)

from sklearn.metrics import accuracy_score # accuracy 점수를 확인
acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", np.round(acc_score, 4))


# MaxAbsScaler
# 훈련에 걸린시간 : 40.62 초
# loss : 0.1220901682972908
# acc : 0.9474
# acc_score : 0.9474

# 훈련에 걸린시간 : 6.4 초
# loss : 0.10129708051681519
# acc : 0.9532
# acc_score : 0.9532


# 훈련에 걸린시간 : 0.12 초
# loss : 0.12287753075361252
# acc : 0.9415
# acc_score : 0.9415