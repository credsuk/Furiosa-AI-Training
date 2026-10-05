# 2026-10-05 AI 학습 평일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-05 (월요일)
- 기준 최신 SHA: `065b917f2019426e281305872a709b7064136add` (`docs: add 2026-10-04 AI review and summary`)
- 학습 코드 기준 SHA: `579b2af291656f6a80cd6959d0a489782b8e54b8` (`day22`)
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)

## 복습 핵심

- `reshape`는 원소 수를 보존하므로 전체 원소 수와 새 축의 곱을 확인합니다.
- pandas 열 순서와 NumPy feature 순서는 모델 입력의 의미를 결정합니다.
- 정수 label·one-hot label·출력 activation·loss는 서로 맞춰야 합니다.
- 모델 실행 결과와 코드 주석의 shape가 다를 수 있으므로 실제 배열을 확인합니다.

## 객관식

### 1. NumPy `reshape`와 원소 수

```python
import numpy as np
x = np.zeros((4, 6))
y = x.reshape(-1, 3, 2)
```

`y.shape`는 무엇인가요?

- **A.** `(4, 3, 2)`
- **B.** `(2, 3, 2)`
- **C.** `(3, 4, 2)`
- **D.** 오류: `reshape`는 3차원으로 바꿀 수 없다.

### 2. pandas `drop`의 축

```python
x = train_csv.drop(["casual", "registered", "count"], axis=1)
```

가장 정확한 설명은 무엇인가요?

- **A.** 지정한 열을 삭제하고 행의 개수는 유지한다.
- **B.** 지정한 위치의 행을 삭제한다.
- **C.** 결측치를 삭제하고 열 이름도 자동 정렬한다.
- **D.** DataFrame을 자동으로 `(batch, timestep, feature)`로 reshape한다.

### 3. sparse label과 분류 loss

10개 class의 정수 label `y`와 `Dense(10, activation="softmax")`를 사용할 때 가장 적절한 조합은 무엇인가요?

- **A.** `sparse_categorical_crossentropy`와 정수 label
- **B.** `binary_crossentropy`와 정수 label
- **C.** `categorical_crossentropy`와 정수 label을 항상 그대로 사용
- **D.** `mse`와 `Dense(1, activation="sigmoid")`

### 4. IMDB 코드의 평가 입력

`keras2/keras62_2_imdb.py`의 다음 코드에 대한 정적 점검으로 가장 적절한 것은 무엇인가요?

```python
loss = model.evaluate(x_train, y_test)
```

- **A.** X와 label의 샘플 수가 같으므로 항상 문제없다.
- **B.** 테스트 평가라면 `x_test, y_test`, 훈련 평가라면 `x_train, y_train`처럼 의도에 맞춰 대응시킨다.
- **C.** `evaluate`는 label을 받지 않으므로 `y_test`를 삭제한다.
- **D.** `x_train`을 `np.argmax`로 바꾸면 해결된다.

## 주관식

### 5. Python·NumPy shape 추적

다음 코드의 `a`, `b`, `c` shape를 계산하고 `c`가 선택하는 축을 설명하세요. 실제 NumPy 실행인지 정적 계산인지도 구분하세요.

```python
import numpy as np
a = np.zeros((5, 4, 2))
b = a.reshape(-1, 2, 2)
c = b[1:, ::2, 0]
```

### 6. pandas에서 LSTM 입력까지

다음 흐름을 설명하세요. `drop(..., axis=1)`, scaler의 `fit_transform`과 `transform`, `(batch, timestep, feature)`의 각 축을 포함하고, 8개 feature를 `(4, 2)`로 reshape할 때 열 순서를 바꾸면 안 되는 이유를 적으세요.

```python
x = train_csv.drop(["casual", "registered", "count"], axis=1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 4, 2)
model.add(LSTM(10, input_shape=(4, 2)))
```

### 7. 코드 주석과 실제 결과 검토

`keras2/keras63_sparse1_minist.py`에서 `model.predict(x_test)`의 결과를 `np.argmax(..., axis=1)`로 바꾼 뒤 기대 shape를 쓰세요. MNIST test 샘플 수와 class 수를 근거로 주석 `(10,) (10000,)`의 문제를 설명하고, 실제 실행 없이 확인할 수 없는 범위를 구분하세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·코드·검증 범위를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
