# keras28_Scaler09_fetch_covtype.py 복사

from sklearn.datasets import fetch_covtype
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
datasets = fetch_covtype()
# print(datasets.DESCR) # 용량이 커서 이거 하면 안됨
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

y = pd.get_dummies(y, dtype=int)
print(x.shape, y.shape) # (581012, 54) (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    random_state=513,
    stratify=y,
)


scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성


#3. 컴파일, 훈련
mcp_path = "./_save/keras31/fatch_covtype/"

start_time = time.time()
model = load_model(mcp_path + "k31_0914_1426_0154-0.3967.keras")
end_time = time.time()


#4. 평가 예측
print("================= keras31_MCP_save_09_fatch_covtype =================")

result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print("acc_score : ", acc_score)
print("걸린시간 :", round(end_time - start_time, 2), "초")


## acc = 0.93 이상


# MaxAbsScaler
# loss : 0.24471095204353333
# acc : 0.9050966501235962
# acc_score :  0.9050966245103371
# 걸린시간 : 353.14 초


# loss : 0.3968348205089569
# acc : 0.8359827399253845
# acc_score :  0.8359827335752101
# 걸린시간 : 63.13 초


# loss : 0.3968348205089569
# acc : 0.8359827399253845
# acc_score :  0.8359827335752101
# 걸린시간 : 0.13 초