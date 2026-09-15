from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

#1. 데이터
# (1000,5,5,1) 데이터를 받는다고 치고 모델 구성하는중


#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5, 5, 1)))
# 10 출력 노드의 수 / (2,2) 필터의 크기 / input_shpae 행무시 열우선, 행의크기를 뺀 나머지를 담는다 (몇장 인지만 삭제한다)

model.add(Conv2D(5, (2, 2)))

model.summary()
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50        
                                                                 
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205       
                                                                 
=================================================================
Total params: 255
Trainable params: 255
Non-trainable params: 0
_________________________________________________________________

(None, 4, 4, 10) 2,2로 짤라서 4, 4 가 나왔고, 10은 아웃 노드의 수
(None, 3, 3, 5)  다시 필터가 지나서 3, 3이 됐다, 5는 아웃 노드의 수
여기서 지금 봐야할건.. 크기가 줄었다
"""


#3. 컴파일, 훈련