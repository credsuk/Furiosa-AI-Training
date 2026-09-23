# keras54_RNN1 카피


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# 데이터가 얼마 없으니 수동으로
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ]) 

y = np.array([4,5,6,7,8,9,10]) # (7,)
# print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1) # 3차원으로 변경
# print(x.shape) # (7, 3, 1)


#2. 모델구성
model = Sequential()
model.add(SimpleRNN(2, input_shape=(3, 1))) 

model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 5)                 35        
                                                                 
 dense (Dense)               (None, 7)                 42        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 85
Trainable params: 85

Param = 연산량이 아니라 웨이트의 수다

[ n, 3, 1 ]
batch_size(행의 수), time_stamp(시 계열), feature(특징) 
time_stamp가 늘어봤자 중복이라 계산되지 않는다

파라미터의 걔수 = units*feature + units*bias * units*units

인터넷 검색
( units 개수 * units 개수 ) + ( feature 수 + units 개수 ) + (bias * unit 개수)

"""
