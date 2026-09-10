import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
path = "./_data/kaggle_santander/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sample_submission.csv", index_col=0,)


# print(train_csv.columns)
"""
Index(['target', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6',
       'var_7', 'var_8',
       ...
       'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
       'var_196', 'var_197', 'var_198', 'var_199'],
      dtype='str', length=201)
"""

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
# print(x.shape, y.shape) # (200000, 200) (200000,)

# 판다스를 넘파이로 바꾸기
# y = np.array(y)
y = y.to_numpy()

ohe = OneHotEncoder(sparse_output=False)
y = y.reshape(-1,1) # 2차원 배열로 만들어 줘야함
y = ohe.fit_transform(y)
# print(y.shape) # (200000, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=5212,
    stratify=y, # y는 어떤 데이터를 대상인지를 사용함 Yes가 아님
)

#2. 모델구척
model = Sequential()
model.add(Dense(7, input_dim=200, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(25, activation='relu')) 
model.add(Dense(13, activation='relu')) 
model.add(Dense(7, activation='relu')) 
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=10000,
    batch_size=791,
    verbose=1,
    validation_split=0.2,
    callbacks=[es],
)
end_time = time.time()


#4. 평가 예측
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", round(result[1], 4))

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)

print("accuracy_score : ", np.round(acc_score, 2))
print("걸린시간 : ", round(end_time - start_time, 2), "초")


## 엑셀 다운
y_pred = model.predict(test_csv)
y_pred = np.argmax(y_pred, axis=1)
print(np.unique(y_pred, return_counts=True))
print(y_pred.shape)

submission['target'] = y_pred
submission.to_csv(path + "submit/submission_0910_1040.csv")

"""
훈련에 걸린 시간 :  51.3484 초
loss : 0.24008747935295105
acc :  0.9125000238418579
acc_score :  0.878105
"""

"""
loss : 0.2397013008594513
acc : 0.912
accuracy_score :  0.91
걸린시간 :  122.76 초
"""