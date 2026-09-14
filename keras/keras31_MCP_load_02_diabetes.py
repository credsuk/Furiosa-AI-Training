"""
keras28_Scaler02_diabetes.py 복사
"""
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time
import datetime

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=34)

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성

#3. 컴파일, 훈련

mcp_path = "./_save/keras31/diabetes/"

start_time = time.time()
model = load_model(mcp_path + "k31_0914_1336_3835-4065.9724.keras")
end_time = time.time()

#4. 평가 예측
print("================= keras31_MCP_save_02_diabetes ================")
print("훈련시간 : ", round(end_time - start_time, 2), "초")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)



"""

Epoch 1754/10000
loss :  3194.597900390625
r2 : 0.3773081007759006
"""

# MaxAbsScaler
# loss :  3008.32666015625
# r2 : 0.41361619613924705

# 훈련시간 :  297.67 초
# loss :  3004.165771484375
# r2 : 0.41442719895737423


# 훈련시간 :  0.11 초
# loss :  3004.165771484375
# r2 : 0.41442719895737423