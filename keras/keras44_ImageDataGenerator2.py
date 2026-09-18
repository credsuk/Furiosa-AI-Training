
import numpy as np
import pandas as pd
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping


# 버전에 따라 다르지만 모두 같음 기본값은 원래 쓰던거
# from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D
# from keras.layers import MaxPool2D, GlobalAveragePooling2D
# from tensorflow.python.keras.layers import MaxPool2D, GlobalAveragePooling2D



#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 파이썬은 "." 이 자료형이 붙으면 형변환이 된다 (실질적인 수치화는 이거 하나)
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = "./_data/image/brain/train/"
path_test = "./_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
    path_train,             # 경로
    target_size=(150,150),  # 
    batch_size=160,          # 자르는 사이즈
    class_mode='binary',    # 이진 분류
    color_mode='grayscale', # 흑백
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),
    batch_size=120,
    class_mode='binary',
    color_mode='grayscale', # 흑백
    # shuffle=True            # 섞는 명령어
)

# print(xy_train[0][0].shape) # (160, 150, 150, 1)
# print(xy_train[0][1].shape) # (160,)

# print(xy_train1[0][0].shape) # (160, 150, 150, 1)
# print(xy_train1[0][1].shape) # (160,)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]


# print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
# print(x_test.shape, y_test.shape) # (120, 150, 150, 1) (120,)


#2. 모델 구축
model = Sequential()
model.add(Conv2D(50, (3, 3), input_shape=(150,150,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(36, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()

# exit()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], )

es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=100,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=64,
    validation_split=0.2,
    callbacks=[es,],
)
end_time = time.time()


# 평가 예측
loss = model.evaluate(x_test, y_test)

print("=================================")
print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)


acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", np.round(acc_score, 4))

# 목표치 : acc_score >> 1.0

# 훈련에 걸린시간 : 187.98 초
# loss :  0.021947139874100685
# acc :  1.0
# acc_score : 1.0
