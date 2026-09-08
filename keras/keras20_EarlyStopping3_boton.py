"""
keras19_overfit3_boston.py 복사
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 500,
    restore_best_weights = True,
)
hist = model.fit(x_train, y_train, 
                 epochs = 70000, 
                 batch_size = 24, 
                 validation_split = 0.2,
                 callbacks = [es],
                )


#4. 평가 예측
loss = model.evaluate(x_test, y_test) # 로스율을 계산
print("loss(mse) : ", loss)

# 평가 지표, 예측
from sklearn.metrics import r2_score, mean_squared_error 
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("R2 :", r2)

mse = mean_squared_error(y_test, y_predict) # 스키드런은 원값과 예측값만 주면된다
print("mse :", mse)


# RMSE함수 정의
def RMSE(y_test, y_predict) : # def로 함수를 정의하고 이름을 뒤에 쓴다 
    return np.sqrt(mean_squared_error(y_test, y_predict)) # sqrt는 루트를 거는 함수


rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"

plt.figure(figsize=(9,6))
plt.plot(hist.history["loss"], c="red", label="loss")
plt.plot(hist.history["val_loss"], c="blue", label="val_loss")

plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("loss")

plt.title("보스턴 loss")

plt.grid()
# plt.show()


"""
이전 값
# loss(mse) :  23.284589767456055
# R2 : 0.7202845708869855

Epoch 7925/70000
loss(mse) :  23.48435401916504
R2 : 0.7178848263322362
mse : 23.484352781936426
RMSE :  4.846065701363987

"""




