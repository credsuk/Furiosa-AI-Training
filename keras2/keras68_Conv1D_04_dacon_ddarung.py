# keras52_opmizer4_dacon_ddarung.py 데이터 복사

import numpy as np # 숫자 쪽에서 아주 강력한 놈
import pandas as pd # numpy로 이루어진 아주 강력한놈
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터
path = "./_data/ddarung/" # 이렇게 사용하는걸 상대 경로라고함

train_csv = pd.read_csv(path + "train.csv", index_col=0, ) 
# print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]

# 제출용 파일
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

########################## 결측치 처리 1. 삭제 ###########################
train_csv = train_csv.dropna()
# print(train_csv.shape) # (1328, 10)

########################## csv를 x와 y로 분리 ##########################
x = train_csv.drop(['count'], axis=1) # axis=축 행=0, 열=1 => count란 열을 삭제할꺼다
y = train_csv['count'] # count 컬럼만 빼겠다 이것도 판다스에서 하는..듯?

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=68)

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



########################## submit 물밑 작업 ###########################
print(test_csv.info())
########################## 결측치 처리 2. 평균치 ###########################
test_csv = test_csv.fillna(test_csv.mean()) 


# print(x_train.shape, x_test.shape) # (996, 9) (332, 9)
x_train = x_train.reshape(-1, 3, 3)
x_test = x_test.reshape(-1, 3, 3)
# print(x_train.shape, x_test.shape) # (996, 3, 3) (332, 3, 3)

#2. 모델구성
model = Sequential()
model.add(Conv1D(16, (3,), input_shape=(3,3)))
model.add(Conv1D(32, (2,), padding='same'))
model.add(Conv1D(62, (2,), padding='same'))
model.add(GlobalAveragePooling1D())
model.add(Dense(5, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 학습
from tensorflow.keras.optimizers import Adam
learning_rate = 0.006
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 20,
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs = 50000, 
                 batch_size = 36, 
                 validation_split = 0.2,
                 callbacks = [es]
                )
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse2 = root_mean_squared_error(y_test, y_predict) # 지원해주는 rmse 사용해보기

print("================= 04_dacon_ddarung ================")
print("훈련 시간 :", round(end_time - start_time, 2), "초")
print("loss :", loss)
print("r2 :", r2)
print("mse :", mse)
print("rmse2 :", rmse2)





# RobustScaler
# 훈련 시간 : 19.17 초
# loss : 2369.961181640625
# r2 : 0.6200291266309885
# mse : 2369.961026647106
# rmse : 48.682245497173874
# rmse2 : 48.682245497173874

# learning_rate = 0.0001
# 훈련 시간 : 60.05 초
# loss : 2504.6904296875
# r2 : 0.5984282174022943
# mse : 2504.6905983067477
# rmse : 50.046884001971065
# rmse2 : 50.046884001971065

# ================= 04_dacon_ddarung ================
# 훈련 시간 : 8.81 초
# loss : 2507.343017578125
# r2 : 0.5980029605881529
# mse : 2507.34301760114
# rmse2 : 50.07337633514581

