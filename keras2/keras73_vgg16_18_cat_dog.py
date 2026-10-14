# keras49_01_내가고양이상인가 복사

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
import numpy as np
import time

#1. 데이터
np_path = "./_data/kaggle_cat_dog_npy/"
x_train = np.load(np_path + "keras45_03_x_train.npy") # x_train
y_train = np.load(np_path + "keras45_03_y_train.npy") # y_train
x_test = np.load(np_path + "keras45_03_x_test.npy") # x_test
y_test =np.load(np_path + "keras45_03_y_test.npy") # y_test
my_image =np.load(np_path + "keras48_my_image.npy")


#2. 모델 구축
from keras.applications import VGG16
#2. 모델
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3),
)

# vgg16.trainable = True # 모든 가중치 학습시킨다는 뜻      # 동결X
vgg16.trainable = False # 특징 추출 가중치는 학습 하지 않음     # 동결O

model = Sequential()
model.add(vgg16)
# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(Dropout(0.3))
# model.add(MaxPool2D())
model.add(Conv2D(24, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()



# #3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], )


es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=10,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=10000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es,],
)
end_time = time.time()


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



# 훈련에 걸린시간 : 767.32 초
# loss :  0.5204530954360962
# acc :  0.752348005771637
# acc_score : 0.7523


# 훈련에 걸린시간 : 179.38 초
# loss :  0.2354884296655655
# acc :  0.8991596698760986
# acc_score : 0.8992