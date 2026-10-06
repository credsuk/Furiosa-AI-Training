# 2026-10-06 AI 학습 평일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-06 (화요일)
- 기준 최신 SHA: `6505ae71ffad9cfe9b6065c4e4de792203331b36` (`docs: add 2026-10-05 AI review and summary`)
- 최신 학습 코드 SHA: `579b2af291656f6a80cd6959d0a489782b8e54b8` (`day22`)
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)

## 복습 핵심

- Python에서는 변수명과 반환값의 의미를 코드 그대로 추적합니다.
- NumPy `reshape`와 CNN의 convolution은 각각 원소 수 보존과 유효 영역 크기 변화를 구분해야 합니다.
- pandas/scaler는 train에만 `fit`하고 test에는 같은 기준으로 `transform`합니다.
- `Embedding`의 index 범위, 모델 출력, label과 loss, `evaluate`의 X/y 샘플 수를 함께 확인합니다.

## 객관식

### 1. Python 변수와 반환값

```python
def add_one(value):
    result = value + 1
    return result

number = 3
answer = add_one(number)
```

`answer`와 `result`에 대한 설명으로 가장 정확한 것은 무엇인가요?

- **A.** `answer`는 4이고, 함수가 끝난 뒤 함수 내부 지역변수 `result`는 바깥에서 직접 사용할 수 없다.
- **B.** `answer`는 3이고, `result`는 자동으로 전역변수가 된다.
- **C.** `answer`는 1이고, `result`는 문자열이다.
- **D.** `add_one`은 반환문이 있으므로 오류가 발생한다.

### 2. NumPy `reshape`와 축

```python
import numpy as np
x = np.arange(24)
y = x.reshape(2, 3, 4)
z = y[:, 1, :]
```

`z.shape`는 무엇인가요?

- **A.** `(2, 4)`
- **B.** `(2, 1, 4)`
- **C.** `(3, 4)`
- **D.** `(24,)`

### 3. pandas와 scaler 적용

훈련·테스트 DataFrame을 같은 수치 scaler로 처리할 때 가장 올바른 흐름은 무엇인가요?

- **A.** `x_train = scaler.fit_transform(x_train)`, `x_test = scaler.transform(x_test)`
- **B.** `x_train = scaler.transform(x_train)`, `x_test = scaler.fit_transform(x_test)`
- **C.** 두 데이터 모두 각각 `fit_transform`하여 각자의 평균과 범위를 사용한다.
- **D.** test 데이터의 label을 삭제하면 scaler의 `fit`이 필요 없다.

### 4. Keras `evaluate` 입력 점검

다음 코드의 정적 점검으로 가장 적절한 것은 무엇인가요?

```python
loss = model.evaluate(x_train, y_test)
```

- **A.** 항상 정상이며 `evaluate`는 X와 y의 샘플 수를 확인하지 않는다.
- **B.** 평가 목적에 맞는 대응 쌍을 사용해야 하므로, 훈련 평가라면 `x_train, y_train`, 테스트 평가라면 `x_test, y_test`를 확인한다.
- **C.** `y_test`를 `np.argmax`하면 X/y 불일치가 자동으로 해결된다.
- **D.** `evaluate`에는 X만 전달하므로 y를 제거해야 한다.

## 주관식

### 5. IMDB 코드의 입력·label·출력 정합성

`keras2/keras62_2_imdb.py`를 기준으로 다음을 답하세요.

1. `imdb.load_data(num_words=2000)`로 얻은 sequence를 `pad_sequences(..., maxlen=700)`한 뒤 `x_train`의 일반적인 shape를 쓰세요.
2. `Embedding(np.max(x_train), 2000)`에서 `input_dim`을 정할 때 확인해야 할 index 범위를 설명하세요.
3. `model.evaluate(x_train, y_test)`와 `model.predict(x_train)` 후 `accuracy_score(y_test, y_predict)`에서 발견되는 X/y 샘플 대응 문제를 각각 설명하세요.
4. 실제 TensorFlow 실행 없이 판단한 부분과 실행이 필요한 부분을 구분하세요.

### 6. CNN shape와 주석 검토

`keras2/keras63_sparse1_minist.py`의 입력 shape가 `(N, 28, 28, 1)`이고 모든 `Conv2D`가 기본 `padding='valid'`라고 가정하세요. 다음 convolution을 순서대로 지난 뒤 shape를 계산하세요.

```python
Conv2D(64, (3,3))
Conv2D(32, (3,3))
Conv2D(32, (2,2))
Conv2D(16, (2,2))
Conv2D(16, (2,2))
Conv2D(8, (5,5))
```

각 단계의 `(height, width, channels)`를 쓰고, 코드 주석의 `(22, 22, 6)`, `(16, 16, 8)` 중 실제 계산과 맞지 않는 부분을 찾으세요. 마지막 `GlobalAveragePooling2D`와 `Dense(10, activation='softmax')` 뒤 예측 및 `argmax(axis=1)`의 기대 shape도 설명하세요.

### 7. pandas feature를 LSTM에 넣는 설계

다음 흐름을 초보자에게 설명하듯 답하세요.

```python
features = train_csv.drop(['casual', 'registered', 'count'], axis=1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 4, 2)
model.add(LSTM(10, input_shape=(4, 2)))
```

`axis=1`, `fit_transform`/`transform`, `(batch, timestep, feature)`의 각 축, 8개 feature의 열 순서를 유지해야 하는 이유, 그리고 이 코드만으로는 확인할 수 없는 사항을 각각 적으세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·코드·검증 범위를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
