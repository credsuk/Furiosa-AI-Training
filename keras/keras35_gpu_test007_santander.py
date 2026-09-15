# keras22_sigmoid_santander.py 복사

import numpy as np
import pandas as pd
import time, datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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


scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential();
model.add(Dense(150, input_dim=200, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(250, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(50, activation="relu"))
model.add(Dense(28, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"],)

es = EarlyStopping(
    monitor="val_acc",
    mode="max",
    patience=50,
    restore_best_weights=True,
)

start_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=100,
          batch_size=791,
          validation_split=0.2,
        #   callbacks=[es, ],
          )
end_time = time.time()


#4. 평가 예측
print("================= keras31_MCP_save_07_santander =================")
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

# submission.to_csv(path + "submit/submission_0908_1730.csv")


# GPU
# 훈련에 걸린 시간 :  50.4651 초
# loss : 0.2555839419364929
# acc :  0.9072750210762024
# acc_score :  0.89927

#CPU
# 훈련에 걸린 시간 :  95.0505 초
# loss : 0.2553367018699646
# acc :  0.9094750285148621
# acc_score :  0.899455

