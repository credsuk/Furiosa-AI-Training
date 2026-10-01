import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time


#1. 데이터
text1 = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."
text2 = "개똥이는 기관사를 좋아힌다. 말똥이는 잘생겼다. 기동이는 마구 마구 더 잘생겼다."

token = Tokenizer()
token.fit_on_texts([text1, text2])
# print(token.word_index)

x = token.texts_to_sequences([text1, text2])
x = np.concatenate([x[0], x[1]]) # 2개의 배치로 나오니 concatenate을 통해서 붙여 줘야한다
# print(x.shape) # (24,)


ohe = OneHotEncoder(sparse_output=False)
x = ohe.fit_transform(x.reshape(-1, 1))
print(x.shape) # (24, 16)

# 원래 y값에 OneHotEncoder을 했는데 지금은 x에 하고 있음


