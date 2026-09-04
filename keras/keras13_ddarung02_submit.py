"""
keras13_ddarung01.py 데이터 복사

"""

import numpy as np # 숫자 쪽에서 아주 강력한 놈
import pandas as pd # numpy로 이루어진 아주 강력한놈
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


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
print(train_csv.shape) # (1328, 10)

########################## csv를 x와 y로 분리 ##########################
# train_csv에 마지막 데이터 통계 데이터를 추출해서 x, y를 만들어야함
x = train_csv.drop(['count'], axis=1) # axis=축 행=0, 열=1 => count란 열을 삭제할꺼다
y = train_csv['count'] # count 컬럼만 빼겠다 이것도 판다스에서 하는..듯?
print(x.shape, y.shape) # (1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=68)


########################## submit 물밑 작업 ###########################
print(test_csv.info())
########################## 결측치 처리 2. 평균치 ###########################
# 체우다 라는 뜻에 fill을 사용한다 fill na 하면 na(결측치)값에 채우겠다
# test_csv에 있는 mean()> 평균치 값을 사용해라
test_csv = test_csv.fillna(test_csv.mean()) 
# print(test_csv.info())
print(test_csv.shape) # (715, 9)



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

################# submission.csv 만들기 // count컬럼에 값을 넣어준다
y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit # count컬럼에 데이터 값을 넣어라

# print(submission_csv)
# print(submission_csv.shape)

# csv로 저장할때는 to_csv 확인해보면 to 클립보드 등등이 있음
# 경로를 잘 지정해 줘야함
submission_csv.to_csv(path + "submit/" + "submit_0904_1408.csv") 


"""
rmse값 29.31283 이하 이어야 1등인데.. 32점이면 100등 안으로..

"""


"""
하이퍼 파라미터 튜닝
1. random_state
2. train_size
3. 레이어의 깊이
4. 노드의 갯수
5. epochs
6. batch_size
"""


"""
# 1차 시도
random : 68
train_size = 0.75
epochs = 500
batch_size = 36
model.add(Dense(9))
model.add(Dense(16))
model.add(Dense(5))
model.add(Dense(2))

# 결과
rmse : 50.51238452579118
r2 : 0.590923205548784
"""

