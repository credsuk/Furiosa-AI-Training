from tensorflow.keras.datasets import reuters # 로이터 통신 데이터
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=2000, # 단어사전의 개수, 빈도수의 개수가 높은 단어 순으로 1000개 뽑겠다.
    # maxlen = 100, # 단어 최대길이 제한 / 최대길이가 100이 넘는건 추출하지 않음
    test_split=0.2, # 테스트 데이터의 수
)

# print(x_train)
# print(x_train.shape, y_train.shape) # (8982,) (8982,)
# print(x_test.shape, y_test.shape) # (2246,) (2246,)

# print(y_train) # [ 3  4  3 ... 25  3 25]
# print(np.unique(y_train)) # 어떤게 있는지 판단 np.unique() / pd.value_counts()
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# y 값만보면.. 다중 분류다
# softmax, cross..., OneHotEncode 사용해야함

# print(type(x_train)) # <class 'numpy.ndarray'>
# print(type(x_train[0])) # <class 'list'> > numpy로 변경해 줘야함
# print(len(x_train)) # 8982
# print(len(x_train[0]), len(x_train[1])) # 87 56

print("뉴스기사의 최대길이 : ", max( len(i)  for i in x_train )) # 2376 파이선 문법.. 장난아니네
print("뉴스기사의 최소길이 : ", min( len(i)  for i in x_train )) #  13
print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/len(x_train)) # 145.5398574927633 / truncate로 잡아놔야함

# 전처리 (패드 시쿼스)
def pad(data) :
    return pad_sequences(
        data, 
        padding='pre', 
        maxlen = 300, 
        truncating='pre' 
    )

x_train_padding = pad(x_train)
x_test_padding = pad(x_test)


# print(x_train_padding)
# print(x_train_padding.shape, x_test_padding.shape) # (8982, 200) (2246, 200)


# y 원핫
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (8982, 46) (2246, 46)



#2. 모델구성
model = Sequential()
model.add(Embedding(np.max(x_train_padding), 1500)) # input_dim, output_dim(차원)
model.add(LSTM(36))
model.add(Dense(369, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(218, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(108, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', 
    mode = "min", 
    patience = 150, 
    restore_best_weights= True, 
)

start_time = time.time()
model.fit(
    x_train_padding, y_train, 
    epochs=5000,
    batch_size = 32,
    validation_split=0.2,
    callbacks=[es],
)
end_time = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test_padding, y_test)

y_predict = model.predict(x_test_padding)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)

print("훈련에 걸린시간 : ", round(end_time-start_time, 2), "초")
print("loss : ", loss[0])
print("acc : ", round(loss[1], 3))
print("acc_score : ", acc_score)



# loss, acc 0.67 이상!! >> 0.75 까지 잡아보자


# 훈련에 걸린시간 :  489.86 초
# loss :  1.219560980796814
# acc :  0.7
# acc_score :  0.7003561887800535
