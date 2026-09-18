
import numpy as np
import pandas as pd
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping


#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,
)


path_train = "./_data/image/cat_dog/training_set/"
path_test = "./_data/image/cat_dog/test_set/"

xy_train = train_datagen.flow_from_directory(
    path_train,             # 경로
    target_size=(100, 100),  # 사이즈를 크지 않게 설정
    batch_size=9000,          # 자르는 사이즈
    class_mode='binary',    # 이진 분류
    color_mode='rgb', # 컬러
    shuffle=True,
)

xy_test = train_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=5000,
    class_mode='binary',
    color_mode='rgb', # 컬러
)

# print(xy_train[0][0].shape) # (8005, 80, 80, 3)
# print(xy_train[0][1].shape) # (8005,)
# print(xy_test[0][0].shape) # (2023, 80, 80, 3)
# print(xy_test[0][1].shape) # (2023,)


x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]


#2. 모델 구축
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(100, 100, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
# Total params: 20,173



#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], )


es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=50,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=44,
    validation_split=0.2,
    callbacks=[es,],
)
end_time = time.time()

weights_path = "./_save/keras44/"
model.save_weights(weights_path + "save3.weights.h5")

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


# 기준값 0.77

# 훈련에 걸린시간 : 1046.67 초
# loss :  0.4531117081642151
# acc :  0.7953534126281738
# acc_score : 0.7954
