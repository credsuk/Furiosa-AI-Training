# 2026-10-07 AI 학습 평일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-07 (수요일)
- 기준 최신 로컬 SHA: `72da0aeefe493c3584cd54d38d2c18da2bb834ae` (`docs: add 2026-10-06 AI review and summary`)
- 최신 학습 코드 SHA: `579b2af291656f6a80cd6959d0a489782b8e54b8` (`day22`)
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)
- 주의: 원격 fetch/push는 SSH 설정 오류로 확인하지 못했습니다.

## 복습 핵심

- Python 함수의 지역변수·반환값과 NumPy 배열의 축을 코드 그대로 추적합니다.
- pandas DataFrame의 행·열을 구분하고, scaler는 train에 `fit`, test에는 같은 기준으로 `transform`합니다.
- sequence token의 index 범위, padding 결과, `Embedding.input_dim`, label/loss/output의 정합성을 함께 확인합니다.
- LSTM 입력은 `(batch, timestep, feature)`이며 표 데이터를 reshape할 때 feature 열 순서가 의미를 가집니다.

## 객관식

### 1. Python 지역변수와 반환값

```python
def make_pair(value):
    doubled = value * 2
    return doubled, value

result = make_pair(4)
```

`result`와 `doubled`에 대한 설명으로 가장 정확한 것은 무엇인가요?

- **A.** `result`는 `(8, 4)`이고, 함수 밖에서 `doubled`라는 지역변수를 직접 사용할 수 없다.
- **B.** `result`는 `8`이고, `doubled`는 자동으로 전역변수가 된다.
- **C.** `result`는 `(4, 8)`이고, `doubled`는 문자열이다.
- **D.** 반환값이 두 개이므로 Python 문법 오류가 발생한다.

### 2. NumPy 다차원 indexing

```python
import numpy as np
x = np.arange(24).reshape(2, 3, 4)
y = x[:, :, 1]
```

`y.shape`는 무엇인가요?

- **A.** `(2, 3)`
- **B.** `(2, 3, 1)`
- **C.** `(3, 4)`
- **D.** `(24,)`

### 3. pandas와 train/test scaler

다음 코드의 목적에 맞는 설명은 무엇인가요?

```python
x_train = train_df.drop(['target'], axis=1)
x_test = test_df.drop(['target'], axis=1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

- **A.** `axis=1`은 행을 삭제하며, 두 데이터에 서로 다른 기준을 적용한다.
- **B.** `axis=1`은 열을 삭제하고, scaler 기준은 train에서 학습한 뒤 test에 재사용한다.
- **C.** `transform`은 scaler 기준을 test 데이터로 다시 학습한다.
- **D.** `fit_transform`은 target 열을 자동으로 예측한다.

### 4. Embedding index 범위

token sequence에 실제로 `0`부터 `1999`까지의 정수 index가 들어 있을 때, 다음 중 가장 안전한 `Embedding` 설정은 무엇인가요?

- **A.** `Embedding(input_dim=1999, output_dim=64)`
- **B.** `Embedding(input_dim=2000, output_dim=64)`
- **C.** `Embedding(input_dim=2001, output_dim=64)`
- **D.** `Embedding(input_dim=64, output_dim=2000)`

## 주관식

### 5. NumPy shape와 LSTM 입력

`x`의 shape가 `(8708, 8)`인 표 데이터에 다음 코드를 적용했습니다.

```python
x = x.reshape(-1, 4, 2)
model.add(LSTM(10, input_shape=(4, 2)))
```

1. reshape 뒤 `x.shape`를 쓰세요.
2. `(batch, timestep, feature)`의 각 축을 이 코드에 대응시키세요.
3. 8개 원래 열의 순서가 바뀌면 왜 모델이 같은 의미의 데이터를 받지 못할 수 있는지 설명하세요.
4. 이 코드만으로 실제 성능이나 학습 성공을 단정할 수 없는 이유를 쓰세요.

### 6. Reuters/IMDB sequence 전처리 검토

`pad_sequences(data, padding='pre', maxlen=300, truncating='pre')`의 역할을 설명하고, `num_words=2000`으로 로드한 sequence를 `Embedding`에 넣을 때 `input_dim`을 정하는 방법을 설명하세요. 또한 `model.evaluate()`에 전달하는 X와 y가 반드시 같은 샘플 수와 대응 관계를 가져야 하는 이유를 쓰세요.

### 7. sparse label과 예측 shape

MNIST처럼 정수 label `y_test.shape == (10000,)`을 사용하는 모델이 `Dense(10, activation='softmax')`를 출력한다고 합시다.

```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
y_predict = model.predict(x_test)
y_class = np.argmax(y_predict, axis=1)
```

`y_predict.shape`와 `y_class.shape`를 쓰고, 왜 `sparse_categorical_crossentropy`에서는 one-hot 변환을 생략할 수 있는지 설명하세요. 실제 TensorFlow 실행 없이 확인할 수 있는 것과 확인할 수 없는 것도 구분하세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·코드·검증 범위를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
