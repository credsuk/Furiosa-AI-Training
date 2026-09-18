# keras28_Scaler07_santander.py 복사

import numpy as np
import pandas as pd
import time
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
path = "c:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0,)
test_csv = pd.read_csv(path + "test.csv", index_col=0,)
submission = pd.read_csv(path + "sample_submission.csv", index_col=0,)


x = train_csv.drop(["target"], axis=1, )
y = train_csv["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5221, stratify=y)


scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (160000, 200) (40000, 200)


x_train = x_train.reshape(-1, 50, 4, 1)
x_test = x_test.reshape(-1, 50, 4, 1)


#2. 모델구성
model = Sequential()
model.add(Conv2D(15, (3,4), padding='same', input_shape=(50,4,1), activation='relu'))
model.add(Conv2D(10, (3,4), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation="sigmoid"))



#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"],)

es = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=200,
    restore_best_weights=True,
)

start_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=500,
          batch_size=791,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()


#4. 평가 예측
print("================= santander =================")
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(test_csv.to_numpy().reshape(-1, 50, 4, 1))
y_pred = np.round(y_pred)
acc_score = accuracy_score(y, y_pred) ####### 여기 좀 이상함 확인해봐야됨 #########


print("훈련에 걸린 시간 : ", np.round(end_time - start_time, 4), "초")
print("loss :", loss[0])
print("acc : ", loss[1])
print("acc_score : ", acc_score)



# MaxAbsScaler
# 훈련에 걸린 시간 :  156.2377 초
# loss : 0.2414882630109787
# acc :  0.9114500284194946
# acc_score :  0.89887

# 훈련에 걸린 시간 :  324.9737 초
# loss : 0.2714565396308899
# acc :  0.9050250053405762
# acc_score :  0.10049
