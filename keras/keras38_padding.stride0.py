import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.preprocessing import OneHotEncoder



#2. 모델구성
model = Sequential()
model.add(
    Conv2D(10, (2,2), 
    input_shape=(10, 10, 1),
    # padding='valid', # 기본값 shape를 변경시킨다
    padding='same', # 빈공간을 0으로 채워서 Output Shape가 변경되지 않는다 
    strides=2, # 1이 디폴트 
    # 겹치는 범위를 조절 2를 주면 안겹치게 할 수 있다(x값에 따라서 달라짐) # 100% 권하지 않음알아서 잘야함
    # 소실나지 않는 선에서 중복하는게 좋다
))


model.add(
    Conv2D(filters=9, kernel_size=(3,3),
    padding='valid',
    # padding='same', 
    strides=2,
))

model.summary()
