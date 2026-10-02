from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, GRU, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=2000, # 단어사전의 개수, 빈도수의 개수가 높은 단어 순으로 1000개 뽑겠다.
    # maxlen = 100, # 단어 최대길이 제한 / 최대길이가 100이 넘는건 추출하지 않음
    # test_split=0.2, # 테스트 데이터의 수
)

print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(np.unique(y_train)) # [0 1] 이진이네

print("최대길이 : ", max( len(i)  for i in x_train )) # 2494 
print("최소길이 : ", min( len(i)  for i in x_train )) #  11
print("평균길이 : ", sum(map(len, x_train))/len(x_train)) # 238.71364

# 전처리 (패드 스쿼드)
def pad(data) :
    return pad_sequences(
        data, 
        padding='pre', 
        maxlen = 700, 
        truncating='pre' 
    )

x_train = pad(x_train)
x_test = pad(x_test)
# print(x_train.shape, x_test.shape) # (25000, 800) (25000, 800)



# print(y_train.shape, y_test.shape) # (25000,) (25000,)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (25000, 2) (25000, 2)


#2. 모델 구성
model = Sequential()
model.add(Embedding(np.max(x_train), 2000))
model.add(LSTM(20))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', 
    mode = "min", 
    patience = 50, 
    restore_best_weights= True, 
)

start_time = time.time()
model.fit(
    x_train, y_train, 
    epochs=100000,
    batch_size = 82,
    validation_split=0.2,
    callbacks=[es],
)
end_time = time.time()



#4. 평가, 예측
loss = model.evaluate(x_train, y_test)

y_predict = model.predict(x_train)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)

print("훈련에 걸린시간 : ", round(end_time-start_time, 2), "초")
print("loss : ", loss[0])
print("acc : ", round(loss[1], 3))
print("acc_score : ", acc_score)



# acc 0.6 이상
# 훈련에 걸린시간 :  729.41 초
# loss :  1.461336374282837
# acc :  0.665
# acc_score :  0.6651825467497774