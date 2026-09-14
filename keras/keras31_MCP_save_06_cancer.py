# keras28_Scaler06_cancer.py 불러옴

import numpy as np
import pandas as pd
import time
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer # load_breast_cancer 유방암관련 데이터셋 / 분류형 데이터 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_breast_cancer()

x = datasets["data"]
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3,
    random_state=124, 
    stratify=y, # y데이터를 stratify해라 골고루 나오게 해라
)

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=30, activation='relu'))
model.add(Dense(29, activation='relu'))
model.add(Dense(41, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 무조건 sigmoid를 넣는다 이거 하나다


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'], # 이렇게 줄여서도 사용 가능함 리스트니깐 나중에 친구들이 나옴, 이 앖이 0과 1로 반올림 해줌
            )

es = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=50,
    restore_best_weights=True
)

############### mcp 세이브 파일명 만들기 시작 ####################
mcp_path = "./_save/keras31/cancer/"
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
          batch_size=46,
          validation_split=0.2,
          callbacks=[es, mcp]
        )
end_time = time.time()



#4. 평가, 예측
print("================= keras31_MCP_save_06_cancer =================")
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
# print(np.round(y_pred[:10]))

rmse = root_mean_squared_error(y_test, y_pred)

print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss :", loss[0])
print("acc :", round(loss[1], 4))
# print("rmse :", rmse)

from sklearn.metrics import accuracy_score # accuracy 점수를 확인
acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", np.round(acc_score, 4))


# MaxAbsScaler
# 훈련에 걸린시간 : 40.62 초
# loss : 0.1220901682972908
# acc : 0.9474
# acc_score : 0.9474

# 훈련에 걸린시간 : 6.4 초
# loss : 0.10129708051681519
# acc : 0.9532
# acc_score : 0.9532