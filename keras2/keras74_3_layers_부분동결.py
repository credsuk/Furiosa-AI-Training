import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(
    weights = 'imagenet',
    include_top=False, 
    input_shape=(32, 32, 3),
)
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100)) 
model.add(Dense(100))
model.add(Dense(100))

# 대체로 하단부분은 동결하지 않는다 왜냐하면 데이터 출력 부분이긴 하다
model.add(Dense(100, trainable=False)) # trainable=False 학습을 시키지 않겠다
model.add(Dense(10, activation='softmax'))

# Flatten도 쳐준다
model.layers[2].trainable = False

# 동결되는 위치를 확인해보자

model.summary()

print(len(model.weights)) # 30 = 16(vgg16) * 2 (w, b) + 2 (dense) + * 2(w,b)
print(len(model.trainable_weights)) 

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
#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True
# 4       Dense    dense_2             True
# 5       Dense    dense_3             True
# 6       Dense    dense_4             True




