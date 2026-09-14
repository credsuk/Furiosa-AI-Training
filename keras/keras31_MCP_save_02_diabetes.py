"""
keras28_Scaler02_diabetes.py 복사
"""
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
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

############### mcp 세이브 파일명 만들기 시작 ####################
mcp_path = "./_save/keras31/diabetes/"
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
                 epochs=10000, 
                 batch_size=75, 
                 validation_split=0.2,
                 callbacks=[es, mcp],
                )

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