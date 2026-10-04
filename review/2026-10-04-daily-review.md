# 2026-10-04 AI 학습 주말 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-04 (일요일)
- 기준 최신 SHA: `159bb0bdfe1bf3b4e59f70181a18f86b2771f1d1` (`docs: add 2026-10-03 AI review and summary`)
- 학습 코드 기준 SHA: `579b2af291656f6a80cd6959d0a489782b8e54b8` (`day22`)
- 문제 구성: 총 10문제 (객관식 5, 주관식 5)

## 복습 핵심

- 정수 class label은 `sparse_categorical_crossentropy`, one-hot label은 `categorical_crossentropy`와 먼저 대응시킵니다.
- `Reshape`는 원소 수를 보존하지만, 축을 timestep·feature로 해석하는 의미까지 자동으로 보장하지는 않습니다.
- `MinMaxScaler`는 train에 `fit`, test에는 같은 scaler로 `transform`만 적용합니다.
- 시계열 평가에서는 미래 정보가 과거 학습에 섞이지 않도록 시간 순서를 보존합니다.
- 코드의 주석·기대 shape·기록된 실행 결과와 현재 코드의 실제 shape를 분리해 확인합니다.

## 객관식

### 1. Python·NumPy reshape와 `-1`

```python
x = np.zeros((6, 4))
y = x.reshape(-1, 2, 2)
```

`y.shape`로 맞는 것은 무엇인가요?

- **A.** `(6, 2, 2)`
- **B.** `(3, 2, 2)`
- **C.** `(2, 2, 6)`
- **D.** 오류: `reshape`에서 `-1`은 사용할 수 없다.

### 2. pandas 열 삭제

```python
x = train_csv.drop(["casual", "registered", "count"], axis=1)
```

이 코드의 설명으로 가장 적절한 것은 무엇인가요?

- **A.** 지정한 세 열을 삭제하고 행은 유지한다.
- **B.** 지정한 세 행을 삭제한다.
- **C.** 결측치를 자동으로 제거한다.
- **D.** DataFrame을 자동으로 `(batch, timestep, feature)` 배열로 변환한다.

### 3. 분류 label·loss·출력

3개 class의 정수 label과 `Dense(3, activation="softmax")`를 사용하는 모델에 적절한 조합은 무엇인가요?

- **A.** `sparse_categorical_crossentropy`와 정수 label
- **B.** `binary_crossentropy`와 반드시 3열 one-hot label
- **C.** `categorical_crossentropy`와 정수 label을 그대로 사용
- **D.** `mse`와 `Dense(1, activation="sigmoid")`

### 4. scaler 적용 순서

훈련 데이터와 테스트 데이터에 `MinMaxScaler`를 적용할 때 올바른 흐름은 무엇인가요?

- **A.** train에 `fit_transform`, test에 같은 scaler의 `transform`
- **B.** train과 test 각각에 `fit_transform`
- **C.** test에만 `fit_transform`하고 train은 원본 사용
- **D.** 두 데이터에 `transform`만 적용하고 `fit`하지 않음

### 5. CNN 출력과 LSTM 입력

`(batch, 23, 23, 16)`을 `Reshape((23 * 23, 16))`한 뒤 `LSTM(10)`에 전달할 때 맞는 설명은 무엇인가요?

- **A.** 입력 shape는 `(batch, 529, 16)`이고 529는 timestep, 16은 feature이다.
- **B.** 입력 shape는 `(batch, 23, 23, 16)` 그대로이며 Reshape는 무시된다.
- **C.** 529는 LSTM hidden unit 수이고 16은 출력 class 수이다.
- **D.** Reshape가 원소를 삭제하여 `(batch, 10)`을 만든다.

## 주관식

### 6. NumPy shape 추적과 indexing

다음 코드의 각 단계 shape를 쓰고, `a[1:, ::2, 0]`이 선택하는 축의 의미와 결과 shape를 설명하세요.

```python
import numpy as np

a = np.zeros((5, 4, 2))
b = a.reshape(-1, 2, 2)
c = b[1:, ::2, 0]
```

실제 NumPy를 실행해 확인한 것인지, shape를 정적으로 계산한 것인지도 구분하세요.

### 7. Python window 함수의 정확성 점검

시계열 길이가 `n`이고 `timestep`이 4일 때 다음 함수의 반환 shape를 설명하고, 실행 오류가 있다면 수정하세요.

```python
def split_xy(data, timestep):
    x, y = [], []
    for i in range(len(data) - timestep):
        window = data[i:i + timestep + 1]
        x.append(window[:-1])
        y.append(window[-1])
    return np.array(x), np.array(y)
```

특히 필요한 import, 변수명 일치 여부, `x`와 `y`의 샘플 수를 포함하세요.

### 8. pandas → NumPy → LSTM 입력 해석

다음 흐름을 초보자에게 설명하세요.

```python
x = train_csv.drop(["casual", "registered", "count"], axis=1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 4, 2)
model.add(LSTM(10, input_shape=(4, 2)))
```

`drop`, `fit_transform`, `transform`, `(batch, timestep, feature)`를 각각 설명하고 원래 열 순서를 임의로 바꾸면 안 되는 이유를 적으세요.

### 9. 분류 코드 정적 검토

`keras2/keras62_2_imdb.py`와 `keras2/keras63_sparse1_minist.py`에서 각각 하나 이상, 총 두 가지 이상의 점검 대상을 찾아 설명하세요. 예시는 다음과 같습니다.

- label 표현과 activation·loss의 정합성
- `Embedding`의 `input_dim`이 token ID 범위를 포함하는지
- `evaluate`에 전달하는 X·y의 첫 축 길이
- `predict` 결과 shape와 주석의 일치 여부

실행하지 않은 학습 성능·데이터 다운로드 결과는 만들지 말고 정적 확인과 실행 확인 불가 영역을 나누어 쓰세요.

### 10. RNN·GRU·LSTM과 시간 순서 평가

기본 RNN, GRU, LSTM이 다음 timestep으로 전달하는 상태를 비교하세요. 특히 LSTM의 `hidden state`·`cell state`와 GRU의 별도 cell state 유무를 설명하고, 시계열 train/validation/test를 시간 순서로 나누어야 하는 이유를 미래 정보 누수와 연결해 적으세요.

## 답변 방법

1~10번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·코드·검증 범위를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
