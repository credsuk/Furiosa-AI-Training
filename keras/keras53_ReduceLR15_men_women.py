# keras46_03_save_npy_men_women +  keras47_03_load_npy_men_women 복사

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import GlobalAveragePooling2D, Flatten, MaxPool2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping



#1. 데이터
path_np = "./_data/kaggle_men_women/"
x = np.load(path_np + "keras51_augment5_men_women_x_train.npy")
y = np.load(path_np + "keras51_augment5_men_women_y_train.npy")
# print(x.shape, y.shape) # (27167, 100, 100, 3) (27167,)

# woman 데이터만 추출
# x_woman = x[np.where(y > 0.0)]
# y_woman = y[np.where(y > 0.0)]
# print(x.shape, x_woman.shape) # (27167, 100, 100, 3) (9489, 100, 100, 3)

# np.where() if 문 y_train보다 0보다 큰 값들을 여기다 넣어라
# 라벨값 즉 y_train값중에 영문법이라 man으로 시작하면 0 w로 시작하면 1이다
x_train_woman = x[np.where(y > 0.0)]
y_trian_woman = y[np.where(y > 0.0)]

print(x_train_woman.shape, y_trian_woman.shape) # (9489, 100, 100, 3) (9489,)
print(np.unique(y_trian_woman, return_counts=True)) # (array([1.], dtype=float32), array([9489], dtype=int64))

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
augment_count = 8000 # 4천개

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
print(x_train.shape, y_train.shape) # (20376, 100, 100, 3) (20376,)

t_end_time = time.time()

#2. 모델 구축
model = Sequential()
model.add(Conv2D(64, (4,4), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D())
# model.add(Conv2D(256, (1,1), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()



#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'], )

from tensorflow.keras.callbacks import ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    factor=0.5 # 나누는 기준 기본값은 0.1
)

es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=50,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=5000,
    batch_size=64,
    validation_split=0.2,
    callbacks=[es, rlr],
)
end_time = time.time()

# model_path = "./_save/keras49/"
# model.save(model_path + "men_woman_save03_model.keras")



#4. 평가 예측
loss = model.evaluate(x_test, y_test)

print("=================================")
print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("증폭 데이터 생성에 걸린시간 :", round(t_end_time - t_start_time, 2), "초")
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", np.round(acc_score, 4))


# save02 : acc_score : 0.8707



# Augment
# 훈련에 걸린시간 : 1558.4 초
# 증폭 데이터 생성에 걸린시간 : 9.07 초
# loss :  0.24435527622699738
# acc :  0.9119552373886108
# acc_score : 0.912


# learning_rate = 0.01
# 훈련에 걸린시간 : 1247.31 초
# 증폭 데이터 생성에 걸린시간 : 9.27 초
# loss :  0.33234307169914246
# acc :  0.8542402982711792
# acc_score : 0.8542



# 훈련에 걸린시간 : 927.36 초
# 증폭 데이터 생성에 걸린시간 : 12.84 초
# loss :  0.4634822905063629
# acc :  0.7753239274024963
# acc_score : 0.7753