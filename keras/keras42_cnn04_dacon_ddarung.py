"""
keras20_EarlyStopping4_dacon_ddarung.py 데이터 복사

"""

import numpy as np # 숫자 쪽에서 아주 강력한 놈
import pandas as pd # numpy로 이루어진 아주 강력한놈
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
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



print(x_train.shape, x_test.shape) # (996, 9) (332, 9)


x_train = x_train.reshape(-1, 3, 3, 1)
x_test = x_test.reshape(-1, 3, 3, 1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(15, (3,3), padding='same', input_shape=(3,3,1), activation='relu'))
model.add(Conv2D(10, (3,3), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 학습
model.compile(loss='mse', optimizer='adam')

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
                #  validation_split = 0.2,
                 callbacks = [es]
                )
end_time = time.time()

#4. 평가 예측
print("================= keras28_Scaler04_dacon_ddarung ================")
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




"""
이후 값 MinMaxScaler
Epoch 6972/50000
loss : 6294.53857421875
r2 : -0.00919003674360197
mse : 6294.537879591999
rmse : 79.33812374635538
rmse2 : 79.33812374635538
"""


