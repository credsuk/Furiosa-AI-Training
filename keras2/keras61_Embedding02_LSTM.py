import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
import time


#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]
# 위에 docs와 비교하면 긍정은 1 부정은 0인 데이터다
labels = np.array([1,1,1, 1,1,0, 0,0,0, 0,0,1, 0,1,0])

# 우리는 이걸 DNN으로 구성할꺼다

token = Tokenizer()
token.fit_on_texts(docs)

x = token.texts_to_sequences(docs)

############### 패딩 ###############
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(
    x, # 타켓할 데이터
    padding='pre', # pre앞을 0으로 채운다 / post 뒤를 0으로 채울때
    maxlen = 5, # 최대길이 세팅
    truncating='pre' # 최대길이가 넘어가면 어디를 짜를지 정한다 / 디폴트가 앞(pre)이다
)

print(padded_x)
# print(padded_x.shape) # (15, 5)

# padded_x = padded_x.reshape(15, 5, 1)

x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, random_state=123, test_size=0.1)

print(x_train.shape, x_test.shape) # (13, 5, 1) (2, 5, 1)
print(y_train.shape, y_test.shape) # (13,) (2,)


#2. 모델 구축
model = Sequential()
model.add(LSTM(20, input_shape=(5,1)))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(82))
model.add(Dense(48))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련 

model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=150,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    # batch_size=1,
    epochs=100000,
    validation_split=0.1,
    callbacks=[es]
)
end_time = time.time()


#4. 모델 예측
eval = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, np.round(y_pred))

x_predict = ["개똥이 잘생겼다"]
x_predict = token.texts_to_sequences(x_predict)
x_predict = pad_sequences(
    x_predict, # 타켓할 데이터
    padding='pre', # pre앞을 0으로 채운다 / post 뒤를 0으로 채울때
    maxlen = 5, # 최대길이 세팅
    truncating='pre' # 최대길이가 넘어가면 어디를 짜를지 정한다 / 디폴트가 앞(pre)이다
)

y_predict = model.predict(x_predict)

print("훈련에 걸린시간 : ", round(end_time - start_time, 2), "초")
print("loss : ", eval)
print("acc : ", acc)
print("y_predict :: ", np.round(y_predict))







