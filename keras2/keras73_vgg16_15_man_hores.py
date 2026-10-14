# keras47_01_load_npy_horse 복사

import numpy as np
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Flatten
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from keras.callbacks import EarlyStopping


import tensorflow as tf
import random
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


#1. 데이터
path_np = "./_data/horse_or_human_npy/"
x = np.load(path_np + "horse_or_human_x_npy.npy")
y = np.load(path_np + "horse_or_human_y_npy.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=512, train_size=0.2,)


#2. 모델 구축
from keras.applications import VGG16
#2. 모델
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(200, 200, 3),
)

# vgg16.trainable = True # 모든 가중치 학습시킨다는 뜻      # 동결X
vgg16.trainable = False # 특징 추출 가중치는 학습 하지 않음     # 동결O

model = Sequential()
model.add(vgg16)
# model.add(Conv2D(32, (2,2), input_shape=(200, 200, 3), activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.3))
# model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], )


es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=30,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es,],
)
end_time = time.time()

weights_path = "./_save/keras47/"
model.save_weights(weights_path + "horse_save3.weights.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

print("=================================")
print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", np.round(acc_score, 4))


# 훈련에 걸린시간 : 202.74 초
# loss :  0.05417007952928543
# acc :  0.9829683899879456
# acc_score : 0.983