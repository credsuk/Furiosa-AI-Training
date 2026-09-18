
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(np.__version__) # 자동완성이 안되니 해본다..

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,         # 파이썬은 "." 이 자료형이 붙으면 형변환이 된다 (실질적인 수치화는 이거 하나)
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range=5,       # 각도 조절(정해진 각도만큼 이미지 회전)
    zoom_range=1.2,         # 확대
    shear_range=0.7,        # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 짜부)
    fill_mode='nearest',    # 채우는 함수 ( 이미지를 옴겼을때 비어져 있는 부분은 근처에 있는 데이터로 채워라)
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# 이렇게 지정하면 하위폴더인 ad, normal은 전부 라벨링 한다
path_train = "./_data/image/brain/train/"
path_test = "./_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
    path_train,             # 경로
    target_size=(200,200),  # 
    batch_size=10,          # 자르는 사이즈
    class_mode='binary',    # 이진 분류
    color_mode='grayscale', # 흑백
    shuffle=True,
)
# Found 160 images belonging to 2 classes.
# 출력을 안해도이렇게 나온다

# 이름부터 flow_from_directory 이 디렉토리에서 불러오겠다
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale', # 흑백
    # shuffle=True            # 섞는 명령어
)
# Found 120 images belonging to 2 classes.

# Iterator 는 그냥 연속된 데이터 어렵게 생각하지 말자
# print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000001A0A9401C00>

# print(xy_train.next()) # 이터레이터의 첫번째
# print(xy_train.next()) # 이터레이터의 두번째
# xy가 모여있는 형태로 8개가 있다

# print(xy_train[0]) # 이건 print(xy_train.next())와 동일하다 첫번째 이터레이터

# print(xy_train[0][0]) # 첫번째 배치의 x 데이터
# print(xy_train[0][1]) # 첫번째 배치의 y 데이터

# 눈으로 보기 힘드니깐 shape
# print(xy_train[0][0].shape) # (10, 200, 200, 1)
# print(xy_train[0][1].shape) # (10,)

# print(xy_train[16][0].shape) # 여기서부터 에러. 이유는 160장이다 배치는 10개니깐

# print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'> tuple은 리스트와 동일하지만 수정이 안됨
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>



#2. 모델 구성


#3 컴파일, 훈련


#4. 평가 예측
