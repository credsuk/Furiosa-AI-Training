# 따르링 데이터 받아온 곳 
# https://dacon.io/competitions/open/235576/overview/description

import numpy as np # 숫자 쪽에서 아주 강력한 놈
import pandas as pd # numpy로 이루어진 아주 강력한놈
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = "./_data/ddarung/"
# pandas에서 지원해주는 옵션이 강력함
# index_col은 index로 걸려있는 칸을 입력해 줄 수 있다. 삭제 시킴
# 이건 통 데이터 인데.. y가 제일 마지막에 있다 그래서 x, y분리 해야함
# pandas.read_csv를 하면 첫번째 라인은 통상적으로 제외시켜 버림
# 이건 훈련용 데이터
train_csv = pd.read_csv(path + "train.csv", index_col=0, ) 
# print(train_csv) # [1459 rows x 10 columns]

# test_csv에는 y값(카운트)가 존재하지 않음
# 이건 submission을 추출하기 위한 데이터 predict를 하기위한 데이터
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]

# id값을 제외하고 전체 Nan값으로 되어 있음, 결측지
# 제출용 파일
submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission_csv) # [715 rows x 1 columns]

# print(train_csv.shape) # (1459, 10)
# print(test_csv.shape) # (715, 9)
# print(submission_csv.shape) # (715, 1)

# print(train_csv.columns) # pandas에서 제공하는 columns
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')

# print(train_csv.info()) # pandas에서 제공하는 info
# Non-Null Count 컬럼은 결측치를 제외한 카운트
#  #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    1459 non-null   int64  
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64


########################## 결측치 처리 1. 삭제 ###########################
train_csv = train_csv.dropna() # 판다스에서 제공하는거로 결측치가 있는 로우 자체를 삭제시켜 버림 마지막 na는 non을 제거하는거
print(train_csv.shape) # (1328, 10)
# test_csv = test_csv.dropna() # test도 결측치가 있지만 삭제시키면 안되기 때문에 다음시간에..

# exit() # 중간에 종료시키는 명령어


########################## csv를 x와 y로 분리 ##########################
# train_csv에 마지막 데이터 통계 데이터를 추출해서 x, y를 만들어야함
x = train_csv.drop(['count'], axis=1) # axis=축 행=0, 열=1 => count란 열을 삭제할꺼다
y = train_csv['count'] # count 컬럼만 빼겠다 이것도 판다스에서 하는..듯?
print(x.shape, y.shape) # (1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=68)


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=9))
model.add(Dense(9))
model.add(Dense(16))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 학습
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=36)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict = model.predict(x_test)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(mse) :
    return np.sqrt(mse)

rmse = RMSE(mse)
print("rmse :", rmse)


# mse : 2531.932488962467
# rmse : 50.318311666454655

# rmse값 29.31283 이하 이어야 1등인데.. 32점이면 100등 안으로..