"""
keras28_Scaler05_kaggle_bike.py 복사
"""

import numpy as np
import pandas as pd
import time
import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
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

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구축

#3. 컴파일, 훈련
mcp_path = "./_save/keras31/kaggle_bike/"

start_time = time.time()
model = load_model(mcp_path + "k31_0914_1410_0422-24207.7363.keras")
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

# 훈련 시간 : 0.13 초
# loss : 34409.52734375
# r2 : -0.06114768981933594
# rmse : 185.49807318743234