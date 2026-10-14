import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

# 이렇게 하는 경우는 거의 없지만 가능하다 정도만 보기위해서 해보는거
model = VGG16(
    weights = 'imagenet',
    include_top=False, 
    input_shape=(32, 32, 3),
)
# model.trainable = False

# 모델의 순서를 지정해서 동결시킬 수 있다
model.layers[11].trainable = False

model.summary()


import pandas as pd
pd.set_option('max_colwidth', None)
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]


# 판다스에서는 시리즈, 데이터 프레임 두가지 종류만 있다
result = pd.DataFrame(
    layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)

print(result)



