
import numpy as np
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping


#1. 데이터
np_path = "./_data/kaggle_cat_dog_npy/"
x_train = np.load(np_path + "keras45_03_x_train.npy") # x_train
y_train = np.load(np_path + "keras45_03_y_train.npy") # y_train
x_test = np.load(np_path + "keras45_03_x_test.npy") # x_test
y_test =np.load(np_path + "keras45_03_y_test.npy") # y_test


#2. 모델 구축
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150, 150, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D())
model.add(Conv2D(24, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
# Total params: 20,173



#3. 컴파일, 훈련
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
    epochs=1000,
    batch_size=32,
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
