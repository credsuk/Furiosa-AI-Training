# keras28_Scaler08_wine.py 복사

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


import numpy as np
import pandas as pd
import time, datetime

#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48]))

y = pd.get_dummies(y, dtype=int)
print(x.shape, y.shape) # (178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=4312,
    stratify=y,
)


scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성


#3. 컴파일, 훈련
mcp_path = "./_save/keras31/wine/"

start_time = time.time()
model = load_model(mcp_path + "k31_0914_1414_0280-0.0031.keras")
end_time = time.time()


#4. 평가 예측
print("================= keras31_MCP_save_08_wine =================")
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print("acc_score : ", acc_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")




# acc = 0.95 이상 합격

# MaxAbsScaler
# loss : 0.10959383845329285
# acc : 0.9722222089767456
# acc_score :  0.9722222222222222
# 걸린시간 :  20.5 초


# loss : 0.05577392131090164
# acc : 0.9722222089767456
# acc_score :  0.9722222222222222
# 걸린시간 :  20.84 초

# loss : 0.03641841560602188
# acc : 0.9722222089767456
# acc_score :  0.9722222222222222
# 걸린시간 :  0.11 초