"""
keras20_EarlyStopping2_diabetes.py 복사
"""
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=34)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train), np.max(x_train)) # 0.0 1.0
# print(np.min(x_test), np.max(x_test)) # -0.09132350124497224 1.2724968314321925


# print(x_train.shape, y_train.shape) # (331, 10) (331,)

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=10))
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 1500,
    restore_best_weights=True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=10000, 
                 batch_size=75, 
                 validation_split=0.2,
                 callbacks=[es],
                )

end_time = time.time()

#4. 평가 예측
print("================= keras28_Scaler02_diabetes ================")
print("훈련시간 : ", round(end_time - start_time, 2), "초")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

## 시각화
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"

plt.figure(figsize=(9,6))
plt.plot(hist.history["loss"], c="red", label="loss")
plt.plot(hist.history["val_loss"], c="blue", label="val_loss")

plt.legend(loc="upper right")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("당뇨병 loss")

plt.grid()
# plt.show()



"""

Epoch 1754/10000
loss :  3194.597900390625
r2 : 0.3773081007759006
"""

"""
MinMaxScaler
Epoch 1911/10000
loss :  3107.7021484375
r2 : 0.3942459105548386
"""

# StandardScaler
# Epoch 4234/10000
# loss :  2982.715576171875
# r2 : 0.4186083004632528


# MaxAbsScaler
# loss :  3008.32666015625
# r2 : 0.41361619613924705


# RobustScaler
# 훈련시간 :  365.96 초
# loss :  2998.767333984375
# r2 : 0.415479451342582