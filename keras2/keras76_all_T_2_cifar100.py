# 경고 프린트를 안보이게 하는 방법
import warnings
warnings.filterwarnings("ignore")

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [
    VGG19(include_top = False, input_shape = (32, 32, 3)),
    ResNet50(include_top = False, input_shape = (32, 32, 3)),
    ResNet101(include_top = False, input_shape = (32, 32, 3)), 
    DenseNet201(include_top = False, input_shape = (32, 32, 3)),
    # InceptionV3(), InceptionResNetV2(), # 개,고양이 / 남자,여자에는 적용
    MobileNetV2(include_top = False, input_shape = (32, 32, 3)),
    EfficientNetB0(include_top = False, input_shape = (32, 32, 3)),
]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
import random
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)
#앞으로 사용하지 않아도 넣어줌
# 연산될때 기본적으로 사용하기 때문에 값으로 넣어줘야함
# 명시되지 않아도 기본으로 넣어줘야함


#1. 데이터
from keras.datasets import cifar10, cifar100
(x_train, y_train), (x_test, y_test)  = cifar100.load_data()

# 스케일링
x_train = x_train/255.
x_test = x_test/255.

for model_i in model_list:
    #2. 모델 구축
    model_i.trainable = False
    
    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(100, activation='softmax'))


    #3. 컴파일 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(
            monitor='val_loss',
        mode="min",
        patience=50,
        restore_best_weights=True,
    )

    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        patience=20,
        verbose=1,
        factor=0.5 # 나누는 기준 기본값은 0.1
    )


    model.fit(x_train, y_train, 
              epochs=10000, 
              batch_size=32,
              verbose=0,
              validation_split=0.2,
              callbacks=[es, rlr],
    )

    #4. 평가 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("=================================================================================")
    print("모델명 : ", model_i.name, "loss : ", loss[0], 'acc : ', round(loss[1], 2))









