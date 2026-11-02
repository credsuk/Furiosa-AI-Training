# 2026-10-25 AI 주말 복습 문제

대상 범위: 2026-10-25 학습 코드 없음 → 최근에 다루지 않은 `keras2/keras60_Tokenizer1.py`·`Tokenizer2.py`, `keras2/keras61_Embedding01_DNN.py`·`02_LSTM.py`·`03_ohe_LSTM.py`·`04_중요.py`, `keras2/keras62_1_reuters.py`·`62_2_imdb.py`(문장 → 숫자 → 벡터)와 Python·NumPy·pandas 기초
비고: 2026-10-25(일)에는 새 학습 코드가 없어, 2026-10-23(트리 모델 평가·병렬 병합)과 2026-10-24(도구 호출·메모리)에서 다룬 주제를 피해 **텍스트 전처리**를 골랐습니다. 문제에 나오는 숫자(단어 번호·shape·확률)는 실제 실행으로 확인한 값입니다.
문제 수: 10문제 (객관식 5문제, 주관식 5문제)

## 객관식

### 1. `keras2/keras61_Embedding01_DNN.py`의 문장 15개를 그대로 넣고 `Tokenizer`를 만들었습니다. 이때 `token.word_index`에서 **1번**을 받는 단어는 무엇일까요?

```python
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
```

- A. `'참'`
- B. `'너무'`
- C. `'재미있다'`
- D. `'개똥이'`

### 2. 아래 코드의 `padded_x[2]` 값은 무엇일까요? (`x`는 위 문장들을 `texts_to_sequences`로 바꾼 결과입니다.)

```python
padded_x = pad_sequences(x, padding='pre', maxlen=5, truncating='pre')
print(padded_x[2])   # '참 잘만든 영화에요'
```

- A. `[0, 0, 1, 5, 6]`
- B. `[1, 5, 6, 0, 0]`
- C. `[0, 0, 0, 1, 5]`
- D. `[1, 5, 6]`

### 3. `keras2/keras61_Embedding04_중요.py`는 `padded_x`의 최댓값이 30일 때 아래처럼 임베딩 층을 만듭니다. 이 모델로 학습(`fit`)하면 어떻게 될까요?

```python
model.add(Embedding(input_dim=np.max(padded_x), output_dim=10, input_length=5))
```

- A. 표가 30행이라 30번 단어까지 문제없이 학습된다
- B. `InvalidArgumentError: indices[...] = 30 is not in [0, 30)` 오류가 난다. `input_dim`은 31이어야 한다
- C. 표가 자동으로 늘어나 31행이 되므로 그냥 학습된다
- D. 30번 단어가 자동으로 0번(빈칸)으로 바뀌어 학습된다

### 4. `keras2/keras61_Embedding02_LSTM.py`에 대한 설명으로 **맞는 것**은 무엇일까요?

```python
padded_x = pad_sequences(x, padding='pre', maxlen=5, truncating='pre')
# padded_x = padded_x.reshape(15, 5, 1)
x_train, x_test, y_train, y_test = train_test_split(padded_x, labels, random_state=123, test_size=0.1)
model.add(LSTM(20, input_shape=(5,1)))
```

- A. `padded_x`는 `(13, 5, 1)`이라 LSTM에 그대로 들어간다
- B. `padded_x`는 `(13, 5)` 2차원이고 reshape 줄이 주석 처리되어 있어, LSTM이 기대하는 3차원과 다르다
- C. LSTM은 2차원 입력만 받으므로 이 코드가 원래 맞다
- D. `pad_sequences`가 자동으로 3차원을 만들어 준다

### 5. 다음 코드의 출력은 무엇일까요?

```python
import numpy as np
a = np.array([[0, 0, 0, 2, 3],
              [0, 0, 1, 5, 6]])
print(a.shape, a.max(), a.sum(axis=1))
```

- A. `(2, 5) 6 [5 12]`
- B. `(2, 5) 3 [5 12]`
- C. `(5, 2) 6 [0 1 6 7 9]`
- D. `(2, 5) 6 [0 0 1 7 9]`

## 주관식

### 6. 새 문장 `"개똥이 잘생겼다"`가 DNN 모델의 입력이 되기까지의 과정을 **순서대로 5단계**로 설명해 보세요.

- 각 단계에서 무엇을 하는지, 값이 어떻게 바뀌는지(`[[24, 27]]` → `[[0, 0, 0, 24, 27]]` 같은 실제 값)를 함께 적어 보세요.
- 마지막에 "왜 길이를 똑같이 맞춰야 하는지"를 자기 말로 한두 문장으로 설명해 보세요.

### 7. 원-핫 인코딩(`to_categorical`)과 `Embedding` 층은 둘 다 단어를 숫자 여러 개로 바꿉니다. 하지만 다른 점이 있습니다.

1. 두 방법의 차이를 **두 가지** 적어 보세요.
2. `Embedding(31, 10)`이면 표의 숫자는 모두 몇 개일까요? 계산 과정도 적어 보세요.
3. `input_dim`을 30으로 쓰면 어떤 문제가 생길까요? (`padded_x`의 최댓값이 30일 때)

### 8. `keras2/keras61_Embedding03_ohe_LSTM.py`는 학습은 되지만 마지막 예측에서 오류가 납니다.

```python
x_predict = pad_sequences(token.texts_to_sequences(["개똥이 잘생겼다"]), maxlen=5)  # [[0,0,0,24,27]]
x_predict = to_categorical(x_predict)
y_predict = model.predict(x_predict)   # ← 여기서 오류
```

1. `to_categorical`의 결과 모양이 `(1, 5, 31)`이 아니라 `(1, 5, 28)`이 되는 이유를 설명해 보세요.
2. 실제로 어떤 오류 메시지가 나오는지 적어 보세요.
3. 이 문제를 고치는 방법을 **두 가지** 제안해 보세요.

### 9. `keras2/keras62_2_imdb.py`에는 고쳐야 할 곳이 두 군데 있습니다. 찾아서 설명해 보세요.

```python
model.add(Dense(2, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
...
loss = model.evaluate(x_train, y_test)
y_predict = model.predict(x_train)
```

1. `Dense(2, activation='sigmoid')` + `binary_crossentropy` 조합은 왜 두 갈래 분류에 어울리지 않을까요? (힌트: 한 행의 두 출력을 더하면 1이 되는지)
2. `evaluate(x_train, y_test)`는 무엇이 잘못되었나요?
3. 두 곳을 각각 올바르게 고쳐 적어 보세요.

### 10. 다음 코드의 세 출력을 적고, `0`이 데이터에서 어떤 역할을 하는지 설명해 보세요.

```python
import numpy as np
import pandas as pd

padded = np.array([[0, 0, 0, 2, 3],
                   [0, 0, 1, 5, 6],
                   [10, 11, 12, 13, 14]])
print(padded.shape, padded.max())
print(padded.sum(axis=1))

words = ['마구', '진짜', '마구', '매우', '마구', '진짜', '매우', '마구']
print(pd.Series(words).value_counts().to_dict())
```

1. 세 `print`의 출력을 각각 적어 보세요.
2. `padded` 안의 `0`은 실제 단어 번호인가요? 아니라면 무엇인가요?
3. `padded.sum(axis=1)`이 세 번째 행에서만 큰 이유를 설명해 보세요.

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요. 계산 문제는 계산 과정을 남겨 주세요.
- 이번 문제는 코드의 흐름과 계산 결과를 묻습니다. 실제로 긴 학습을 돌린 정확도나 reuters/imdb 전체 학습 결과를 묻는 문제는 없습니다.
- 1·2·3·5·8·10번의 값과 오류 메시지는 실제 실행(tensorflow-cpu 2.15.1, keras 2.15.0, numpy 1.26.4, pandas 3.0.6)으로 확인했습니다.
