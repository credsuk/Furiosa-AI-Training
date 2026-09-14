"""
keras28_Scaler04_dacon_ddarung.py 데이터 복사

"""

import numpy as np # 숫자 쪽에서 아주 강력한 놈
import pandas as pd # numpy로 이루어진 아주 강력한놈
import time
import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터
path = "./_data/ddarung/" # 이렇게 사용하는걸 상대 경로라고함

train_csv = pd.read_csv(path + "train.csv", index_col=0, ) 
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)

# 결측치 처리
train_csv = train_csv.dropna() # 삭제
test_csv = test_csv.fillna(test_csv.mean()) # 평균치

x = train_csv.drop(['count'], axis=1) # axis=축 행=0, 열=1 => count란 열을 삭제할꺼다
y = train_csv['count'] # count 컬럼만 빼겠다 이것도 판다스에서 하는..듯?

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=68)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(3, activation='relu', input_dim=9))
model.add(Dense(9, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 학습
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 20,
    restore_best_weights = True,
)

############### mcp 세이브 파일명 만들기 시작 ####################
mcp_path = "./_save/keras31/dacon_ddarung/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
filename = "{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([mcp_path, "k31_", date, "_", filename])
############### mcp 세이브 파일명 만들기 끝 ####################

mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto', 
    save_best_only=True, # 최고의 값만 저장해라
    filepath=filepath, # 파일 경로
    verbose=1, # 보여줄지 안보여줄지 확인하는 항목
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs = 50000, 
                 batch_size = 36, 
                 validation_split = 0.2,
                 callbacks = [es, mcp]
                )
end_time = time.time()

#4. 평가 예측
print("================= keras31_MCP_save_04_dacon_ddarung ================")
print("훈련 시간 :", round(end_time - start_time, 2), "초")
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(mse) :
    return np.sqrt(mse)

rmse = RMSE(mse)
print("rmse :", rmse)

rmse2 = root_mean_squared_error(y_test, y_predict) # 지원해주는 rmse 사용해보기
print("rmse2 :", rmse2)



# MaxAbsScaler
# 훈련 시간 : 10.0 초
# loss : 2452.15185546875
# r2 : 0.6068516396855697
# mse : 2452.151880416743
# rmse : 49.51920718687591
# rmse2 : 49.51920718687591


# 훈련 시간 : 14.61 초
# loss : 2366.52587890625
# r2 : 0.6205798599362702
# mse : 2366.5259831712165
# rmse : 48.64695245512525
# rmse2 : 48.64695245512525