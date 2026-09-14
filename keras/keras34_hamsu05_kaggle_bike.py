"""
keras28_Scaler05_kaggle_bike.py 복사
"""

import numpy as np
import pandas as pd
import time
import datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(train_csv.shape, test_csv.shape, submission_csv.shape) # (10886, 11) (6493, 8) (6493, 2)

x = train_csv.drop(["casual", "registered", "count"], axis=1)
y = train_csv['count']
# print(x.shape, y.shape) # (10886, 8) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5437089)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구축
# model = Sequential()
# model.add(Dense(3, activation='relu', input_dim=8))
# model.add(Dropout(0.2))
# model.add(Dense(9, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(12, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(7, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(8,)) 
dense1 = Dense(3, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(9, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(12, activation='relu')(drop2)
drop3 = Dropout(0.25)(dense3)
dense4 = Dense(10, activation='relu')(drop3)
dense5 = Dense(7, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 50,
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs = 30000, 
                 batch_size = 32, 
                 validation_split = 0.2,
                 callbacks = [es, ]
                )
end_time = time.time()

#4. 평가 예측
print("================= keras31_MCP_save_05_kaggle_bike =================")
print("훈련 시간 :", round(end_time - start_time, 2), "초")
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse :", rmse)


# MaxAbsScaler
# 훈련 시간 : 424.4 초
# loss : 23305.384765625
# r2 : 0.28129082918167114
# rmse : 152.66099340122872


# 훈련 시간 : 121.09 초
# loss : 23699.583984375
# r2 : 0.2691340446472168
# rmse : 153.94669202154037


# 훈련 시간 : 22.5 초
# loss : 23771.857421875
# r2 : 0.2669052481651306
# rmse : 154.18124860655072


# 훈련 시간 : 18.97 초
# loss : 24821.298828125
# r2 : 0.234541654586792
# rmse : 157.54777921118088