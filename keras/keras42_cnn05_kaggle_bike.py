"""
keras20_EarlyStopping5_kaggle_bike.py 복사
"""

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
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

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape) # (8708, 8) (2178, 8)


x_train = x_train.reshape(-1, 4, 2, 1)
x_test = x_test.reshape(-1, 4, 2, 1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(15, (2,2), padding='same', input_shape=(4,2,1), activation='relu'))
model.add(Conv2D(10, (2,2), padding='same', activation='relu'))
model.add(Conv2D(5, (2,2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 100,
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs = 500, 
                 batch_size = 32, 
                #  validation_split = 0.2,
                #  callbacks = [es]
                )
end_time = time.time()

#4. 평가 예측
print("================= keras28_Scaler05_kaggle_bike =================")
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


"""
이전 값
Epoch 2535/30000
loss : 22766.14453125
r2 : 0.2979201674461365
rmse : 150.88454039844507
"""

"""
이후 값 MinMaxScaler
Epoch 9868/30000
loss : 21592.708984375
r2 : 0.3341073989868164
rmse : 146.94458458037846
"""

# 훈련 시간 : 49.62 초
# loss : 23671.75390625
# r2 : 0.26999229192733765
# rmse : 153.85627678534925