
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


# #2. 모델 구축
# model = Sequential()
# model.add(Conv2D(32, (3,3), input_shape=(150, 150, 3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(Dropout(0.3))
# model.add(MaxPool2D())
# model.add(Conv2D(24, (3,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # model.summary()
# # # Total params: 20,173



# # #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], )


# es = EarlyStopping(
#     monitor='val_loss',
#     mode="min",
#     patience=10,
#     restore_best_weights=True,
# )

# start_time = time.time()
# model.fit(
#     x_train, y_train,
#     epochs=10000,
#     batch_size=32,
#     validation_split=0.2,
#     callbacks=[es,],
# )
# end_time = time.time()

weights_path = "./_save/keras44/"
# model.save_weights(weights_path + "save4444.weights.h5")
# model.save(weights_path + "cat_save01_model.keras")

model = load_model(weights_path + "cat_save01_model.keras")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

print("=================================")
# print("훈련에 걸린시간 :", round(end_time - start_time, 2), "초")
# print("loss : ", loss[0])
# print("acc : ", loss[1])

# y_pred = model.predict(x_test)
# y_pred = np.round(y_pred)

# acc_score = accuracy_score(y_test, y_pred)
# print("acc_score :", np.round(acc_score, 4))

from tensorflow.keras.preprocessing.image import load_img, img_to_array
my_photo = "./_data/image/cat3.jpg"
img = load_img(my_photo, target_size=(150,150)) # 이미지 한장짜리 불러올때 편하다
arr = img_to_array(img) 
arr = np.expand_dims(arr, axis=0) # reshape랑 동일 차원을 확장하겠다, 0번째에 넣겠다


print("=================================")
arr = arr/255.0
y_predict = model.predict(arr)
# 0이 고양이 1개 강아지
print(y_predict)

# print( round(y_predict, 2) * 100, "% 확율로", y_predict > 0.5 ? "강아지" : "고양이", "입니다")




# 훈련에 걸린시간 : 767.32 초
# loss :  0.5204530954360962
# acc :  0.752348005771637
# acc_score : 0.7523