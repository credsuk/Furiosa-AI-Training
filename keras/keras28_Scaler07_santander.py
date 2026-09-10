# keras22_sigmoid_santander.py 복사

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = "c:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0,)
test_csv = pd.read_csv(path + "test.csv", index_col=0,)
submission = pd.read_csv(path + "sample_submission.csv", index_col=0,)

# print(train_csv.info())
# print(train_csv.describe())
# print(train_csv.isna().sum())


x = train_csv.drop(["target"], axis=1, )
y = train_csv["target"]

# print(x.shape, y.shape) # (200000, 200) (200000,)
# print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=234, stratify=y)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential();
model.add(Dense(5, input_dim=200, activation="relu"))
model.add(Dense(17, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(6, activation="relu"))
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
          epochs=512412512,
          batch_size=791,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()


#3 평가 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(test_csv)
y_pred = np.round(y_pred)
acc_score = accuracy_score(y, y_pred) ####### 여기 좀 이상함 확인해봐야됨 #########


print("훈련에 걸린 시간 : ", np.round(end_time - start_time, 4), "초")
print("loss :", loss[0])
print("acc : ", loss[1])
print("acc_score : ", acc_score)

# print(submission.shape, y_pred.shape)

## 엑셀 다운
submission['target'] = y_pred
print(np.unique(y_pred, return_counts=True))

submission.to_csv(path + "submit/submission_0908_1730.csv")

"""
훈련에 걸린 시간 :  51.3484 초
loss : 0.24008747935295105
acc :  0.9125000238418579
acc_score :  0.878105
"""