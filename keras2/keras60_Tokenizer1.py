# 임베딩

from tensorflow.keras.preprocessing.text import Tokenizer

text = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."

# 우리가 클래스를 정의한 놈을 통상적으로 객체라고 부른다 영어로 인스턴스
# 인스턴스(객체) = 클래스(), 인스턴스 생성
# 함수와 메서드의 차이 
token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) # 중복되는 단어가 제일 앞으로 나온다
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}

print(token.word_counts) # 카운트
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text]) # x로 인스턴스화 시켰습니다
print(x)
# [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
# 나는, 지금, 진짜, 진짜, 매우, 매우 ... 이렇게된거다
# 그래서 이건 원핫인코딩을 통해 위치정보로 변겨을 시켜야 한다

# 원핫 인코딩
import numpy as np
x = np.array(x)
x = x.reshape(-1)
print(x.shape) # (14,)

#1. pandas
# import pandas as pd
# x = pd.get_dummies(x, dtype=int)
# print(x.shape) # (14, 9)

# #2. sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
x = ohe.fit_transform(x.reshape(-1, 1))
print(x.shape) # (14, 9)

# #3. keras / to_categorical
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x) # 카테고리 화 시키는 함수로 panda에도 있다 우선은 이거
# print(x.shape) # (14, 10)
