import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(
    weights = 'imagenet', # 모델의 가중치(파라미터)를 어떤 데이터로 미리 학습된 상태로 가져올지 지정
    #  1,000개의 클래스, 수백만 장의 이미지로 구성된 대규모 데이터셋인 ImageNet에서 이미 학습을 마친 가중치(Pre-trained weights)를 다운로드하여 모델에 적용

    include_top=False, # VGG16 신경망 구조에서 최상단에 위치한 전결합층(Fully Connected Layer, 분류기)을 포함할지 여부
    # False로 설정하면 이미지를 분류하는 마지막 Dense 레이어들을 제외하고, 특징 추출(Feature Extraction)을 담당하는 합성곱층(Convolutional Layers)만 가져옵
    # 전이 학습(Transfer Learning)을 할 때 주로 사용

    input_shape=(32, 32, 3),
)
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

# VGG16 레이어들의 가중치 + 새로 추가한 Dense 레이어의 가중치가 모두 출력
print(len(model.weights)) # 30 = 16(vgg16) * 2 (w, b) + 2 (dense) + b

# 학습되어 가중치 변경될 웨이트 수 
print(len(model.trainable_weights)) 