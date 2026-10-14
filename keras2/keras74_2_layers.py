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
model.add(Dense(10, activation='softmax'))

model.summary()

print(len(model.weights)) # 30 = 16(vgg16) * 2 (w, b) + 2 (dense) + * 2(w,b)
print(len(model.trainable_weights)) 

import pandas as pd
pd.set_option('max_colwidth', None)
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]

print(layers)
# [('Functional', 'vgg16', False), ('Flatten', 'flatten', True), ('Dense', 'dense', True), ('Dense', 'dense_1', True)]

# 판다스에서는 시리즈, 데이터 프레임 두가지 종류만 있다
result = pd.DataFrame(
    layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)

print(result)
# [('Functional', 'vgg16', False), ('Flatten', 'flatten', True), ('Dense', 'dense', True), ('Dense', 'dense_1', True)]
#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True




