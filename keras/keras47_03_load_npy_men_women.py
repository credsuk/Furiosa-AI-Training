
import numpy as np
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Flatten
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping



#1. 데이터
np_path = "./_data/kaggle_men_women/"
x = np.load(np_path + "keras46_03_x_train.npy") # x_train
y = np.load(np_path + "keras46_03_y_train.npy") # y_train

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=421)

print(x_train.shape, x_test.shape) # (6750, 100, 100, 3) (2250, 100, 100, 3)
print(x_train.shape, y_test.shape) # (6750,) (2250,)

#2. 모델 구축
model = Sequential()
model.add(Conv2D(64, (4,4), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D())
model.add(Conv2D(256, (1,1), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


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
    epochs=10000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es,],
)
end_time = time.time()

model_path = "./_save/keras49/"
model.save(model_path + "men_woman_save03_model.keras")



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



# save02 : acc_score : 0.8707