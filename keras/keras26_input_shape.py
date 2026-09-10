# keras23_softmax1_OneHot_iris 복사
import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris # 꽃에 줄기 잎사귀 등등을 가지고 어떤 꽃인지 맞춰야 하는 데이터
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target
# print(pd.DataFrame(y).value_counts()) # 이코드만 봐도 분류구나 생각하면됨 더 많이 쓰인다
"""
[0,0,1,1,2,2] 이렇게 되어있는 (6,)
y 데이터를 이렇게 수정해야함
왜냐하면 1은 2의 2배가 아니기 때문임
->
[ [1,0,0]
  [1,0,0]
  [0,1,0]
  [0,1,0]
  [0,0,1]
  [0,0,1] ] (6,3)

가치가 변하는게 아니라 위치값으로 변경한거라 데이터 변경을 한건 아님
[1, 1, 0] 으로 나오지 않음 이유는 한정함수 sofmax라서 그럼
모든값을 더했을때 1 이상을 넘지 않음
"""

############# OneHot3. sklearn fit_transform #############
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

y = y.reshape(-1,1) # 2차원 배열로 만들어 줘야함
y = ohe.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=532,
    stratify=y, # y는 어떤 데이터를 대상인지를 사용함 Yes가 아님
)


#2. 모델 구축
model = Sequential()
model.add(Dense(7, input_shape=(4, ),  activation='relu')) # input_shape는 shape형태로 넣어야함
model.add(Dense(11, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(13, activation='relu')) # relu는 활성화 함수
model.add(Dense(3, activation='softmax')) # softmax 모두 더해서 엔빵해라 절대 1일 넘지 않고 제일 큰놈을 1로주고 나머지 0

"""
원데이터 > input_shape
(n, 4) -> (4, )
(n, 100, 3) -> (100, 3)
(n, 1000, 100, 3) -> (1000, 100, 3)
"""

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # categorical 범주, # crossentropy > 통상적으로 이거 아니면 이거 이런 형태

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=10000,
    batch_size=8,
    verbose=1,
    validation_split=0.2,
    callbacks=[es],
)
end_time = time.time()


#4. 평가 예측
result = model.evaluate(x_test, y_test)
print("loss : ", result[0])
print("acc : ", round(result[1], 2))

y_predict = model.predict(x_test) # predict 예측 또는 추론

# 가장높은 값을 1로 처리하는게 필요
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score : ", np.round(acc_score, 2))
print("걸린시간 : ", round(end_time - start_time, 2), "초")


# loss :  0.08837046474218369
# acc :  0.93
# accuracy_score :  0.93
# 걸린시간 :  6.68 초

