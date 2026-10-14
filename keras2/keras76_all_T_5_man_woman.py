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
    VGG19(include_top = False, input_shape = (100, 100, 3)),
    ResNet50(include_top = False, input_shape = (100, 100, 3)),
    ResNet101(include_top = False, input_shape = (100, 100, 3)), 
    DenseNet201(include_top = False, input_shape = (100, 100, 3)),
    InceptionV3(), InceptionResNetV2(), # 개,고양이 / 남자,여자에는 적용
    MobileNetV2(include_top = False, input_shape = (100, 100, 3)),
    EfficientNetB0(include_top = False, input_shape = (100, 100, 3)),
]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import random
import time

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)
#앞으로 사용하지 않아도 넣어줌
# 연산될때 기본적으로 사용하기 때문에 값으로 넣어줘야함
# 명시되지 않아도 기본으로 넣어줘야함


#1. 데이터
path_np = "./_data/kaggle_men_women/"
x = np.load(path_np + "keras51_augment5_men_women_x_train.npy")
y = np.load(path_np + "keras51_augment5_men_women_y_train.npy")
# print(x.shape, y.shape) # (27167, 100, 100, 3) (27167,)

x_train_woman = x[np.where(y > 0.0)]
y_trian_woman = y[np.where(y > 0.0)]

x_train, x_test, y_train, y_test  = train_test_split(x, y, random_state=5122)


t_start_time = time.time()
################ 데이터 증폭 #####################
datagen = ImageDataGenerator(
    rescale=1./255,         # 파이썬은 "." 이 자료형이 붙으면 형변환이 된다 (실질적인 수치화는 이거 하나)
    width_shift_range=0.1,  # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=6,       # 각도 조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.2,         # 확대
    fill_mode='nearest',    # 채우는 함수 ( 이미지를 옴겼을때 비어져 있는 부분은 근처에 있는 데이터로 채워라)
)

# augment_count = 40000 # 4만개
augment_count = 8000 # 8천개

randidx = np.random.randint(x_train_woman.shape[0], size=augment_count) # int형으로 랜덤으로 추출한다

# 혹시 메모리 연동으로 인해 원 값이 변경될 수 있으니 .copy()로 확실하게 변경해준다
x_augmented = x_train_woman[randidx].copy()
y_augmented = y_trian_woman[randidx].copy()

# 이렇게 사용하면 위에서 수정하면 아래쪽에서는 쭉~ 수정이 된다
x_augmented = x_augmented.reshape(
    x_augmented.shape[0], # 40000
    x_augmented.shape[1], # 28
    x_augmented.shape[2], # 28
    3
)

x_augmented = datagen.flow(
    x_augmented, 
    y_augmented, 
    batch_size=augment_count,
    shuffle=False,

).next()[0] # x값만 필요하고 y값은 필요 없으니 첫번째 값만 뽑는다

#### 변환 완료 ###
# print(x_augmented.shape) (1, 100, 100, 3)
x_train = x_train.reshape(20375, 100, 100, 3)
y_train = y_train.reshape(20375, )

# np.concatenate() 사슬처럼 역다 / 다차원도 같이 합쳐준다
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape) # (20376, 100, 100, 3) (20376,)

t_end_time = time.time()

for model_i in model_list:
    #2. 모델구축
    model_i.trainable = False
    
    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(3, activation='softmax'))


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









