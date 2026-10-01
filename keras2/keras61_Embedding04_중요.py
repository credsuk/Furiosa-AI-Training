import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
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
    x, 
    padding='pre', 
    maxlen = 5, 
    truncating='pre' 
)

# print(padded_x)
# print(padded_x.shape) # (15, 5)
# print(np.max(padded_x)) # 30
# print(padded_x.shape[1]) # 5


#2. 모델 구축
# 텐서플로는 Embedding 레이어 제공해줌
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

model = Sequential()

############# 임베딩 1 ###############
model.add(Embedding(
    # input_dim=30, # 단어사전의 갯수 / 크게 작성해도 알아서 맞춤  /  입력 데이터의 단어 집합(vocabulary) 크기 
    input_dim=np.max(padded_x), 
    output_dim=10, # 차원 / 임베딩 벡터의 차원 크기
    # input_length=5, # 쉐이프 / 입력 시퀀스의 고정 길이 / 갯수가 다르면 안됨
    input_length=padded_x.shape[1], # 
    ))

model.add(SimpleRNN(10))
model.add(Dense(30))
model.add(Dense(1, activation='sigmoid'))
# model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 5, 10)             300       
                                                                 
 simple_rnn (SimpleRNN)      (None, 10)                210     
                                                                 
=================================================================
"""


############# 임베딩 2 ###############
model.add(Embedding(
    input_dim=30, # 단어사전의 갯수 / 크게 작성해도 알아서 맞춤  /  입력 데이터의 단어 집합(vocabulary) 크기 
    output_dim=10, # 차원 / 임베딩 벡터의 차원 크기
    # input_length=5, # 안해도 알아서 쉐이프를 맞춰 준다
    ))
model.add(Embedding(np.max(padded_x), 10,)) # input_dim, output_dim 이렇게 가능
model.add(SimpleRNN(10))
model.add(Dense(30))
model.add(Dense(1, activation='sigmoid'))


############# 임베딩 3 ###############
model.add(Embedding(30, 10,)) # input_dim, output_dim 이렇게 가능
# model.add(Embedding(30, 10, 5)) # 안됨
# model.add(Embedding(30, 10, input_length=5)) # 가능
model.add(SimpleRNN(10))
model.add(Dense(30))
model.add(Dense(1, activation='sigmoid'))






model.summary()
exit()

#3 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(padded_x, labels, epochs=100)


#4. 평가 예측
evl = model.evaluate(padded_x, labels)
print("loss :: ", evl[0])
print("acc :: ", evl[1])






