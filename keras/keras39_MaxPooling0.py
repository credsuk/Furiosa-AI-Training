import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder



#2. 모델구성
model = Sequential()
model.add(
    Conv2D(10, (2,2), 
    input_shape=(10, 10, 1),
    padding='same', # 빈공간을 0으로 채워서 Output Shape가 변경되지 않는다 
    strides=1,
))

# 통상적으로 Conv2D 뒤에 사용
# MaxPooling를 사용하면 반땡 된다 (당분간은 이렇게 생각해라)
model.add(MaxPooling2D())

model.add(
    Conv2D(filters=9, kernel_size=(3,3),
    padding='valid',
    # padding='same', 
    strides=1,
))

model.summary()
