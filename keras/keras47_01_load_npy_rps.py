
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
path_np = "./_data/rps_npy/"
x = np.load(path_np + "rps_x_npy.npy")
y = np.load(path_np + "rps_y_npy.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.2,)


#2. 모델 구축
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100, 100, 3), activation='relu'))
model.add(Dropout(0.2))
# model.add(MaxPool2D())
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(24, (2,2), activation='relu'))
model.add(Conv2D(24, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(12, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

# exit()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'], )


es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=200,
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
model.save_weights(weights_path + "rsp_save2.weights.h5")

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


# 훈련에 걸린시간 : 56.63 초
# loss :  0.10811012238264084
# acc :  0.9829164147377014
# acc_score : 0.9823


# 훈련에 걸린시간 : 290.75 초
# loss :  0.10486448556184769
# acc :  0.9859670400619507
# acc_score : 0.986

