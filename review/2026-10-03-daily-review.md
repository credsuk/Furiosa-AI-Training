# 2026-10-03 AI 학습 주말 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-03 (토요일)
- 기준 최신 SHA: `d2b288aa562307c3339e0995e0d4aa9dc8d0107a` (`docs: record 2026-10-02 verification results`, 20:06 KST)
- 학습 코드 기준: `579b2af291656f6a80cd6959d0a489782b8e54b8` (`day22`)
- 문제 구성: 총 10문제 (객관식 5, 주관식 5)

## 복습 핵심

- 정수 class label에는 `sparse_categorical_crossentropy`, one-hot label에는 `categorical_crossentropy`를 우선 대응시킵니다.
- CNN 이미지 입력은 일반적으로 `(batch, height, width, channel)`이며, LSTM·GRU 시계열 입력은 `(batch, timestep, feature)`입니다.
- `reshape`는 전체 원소 수를 보존하고 `-1`은 나머지 축으로부터 자동 계산되는 한 축입니다. 모양만 바뀌므로 timestep·feature의 의미는 작성자가 확인해야 합니다.
- `scaler.fit`은 train에만 적용하고 test에는 같은 scaler의 `transform`만 적용합니다.
- 코드의 주석·예상 shape와 실제 실행 결과를 구분합니다. 이번 복습에서는 외부 데이터 다운로드와 모델 학습을 실행하지 않았습니다.

## 객관식

### 1. Python·NumPy `reshape`

```python
x = np.zeros((12, 2))
y = x.reshape(3, -1, 2)
```

`y.shape`는 무엇인가요?

- **A.** `(3, 4, 2)`
- **B.** `(3, 2, 4)`
- **C.** `(12, 2, 1)`
- **D.** 오류: `-1`은 `reshape`에서 사용할 수 없다.

### 2. pandas 열 삭제

다음 코드의 설명으로 가장 적절한 것은 무엇인가요?

```python
x = train_csv.drop(["casual", "registered", "count"], axis=1)
```

- **A.** 지정한 세 열을 삭제하고 행은 삭제하지 않는다.
- **B.** 지정한 세 행을 삭제한다.
- **C.** DataFrame을 자동으로 NumPy 3차원 배열로 바꾼다.
- **D.** 결측치를 모두 0으로 바꾼다.

### 3. scaler 사용 순서

훈련 데이터와 테스트 데이터에 표준화를 적용할 때 올바른 흐름은 무엇인가요?

- **A.** train에 `fit_transform`, test에 같은 scaler로 `transform`을 적용한다.
- **B.** train과 test 각각에 `fit_transform`을 적용한다.
- **C.** test에만 `fit_transform`을 적용하고 train은 원본을 사용한다.
- **D.** 두 데이터에 `transform`만 적용하고 scaler는 학습하지 않는다.

### 4. label·loss·출력

3개 class의 정수 label과 `Dense(3, activation="softmax")`를 사용하는 모델에 가장 적절한 조합은 무엇인가요?

- **A.** `sparse_categorical_crossentropy`와 정수 label
- **B.** `binary_crossentropy`와 3개 확률 출력
- **C.** `categorical_crossentropy`와 반드시 문자열 label
- **D.** `mse`와 `Dense(1, activation="sigmoid")`

### 5. CNN과 LSTM 연결

합성곱 결과가 `(batch, 23, 23, 16)`이고 다음 코드가 실행된다면 올바른 설명은 무엇인가요?

```python
x = Reshape((23 * 23, 16))(x)
x = LSTM(10)(x)
```

- **A.** LSTM 입력은 `(batch, 529, 16)`이며 529는 timestep, 16은 feature이다.
- **B.** LSTM 입력은 `(batch, 23, 23, 16)` 그대로이고 Reshape는 무시된다.
- **C.** 529는 LSTM hidden unit 수이고 feature는 10이다.
- **D.** Reshape가 원소를 삭제하므로 입력은 `(batch, 10)`이다.

## 주관식

### 6. shape 추적

다음 코드의 각 단계 shape를 적고, `-1`이 자동 계산하는 값과 `(batch, timestep, feature)`의 각 축 의미를 설명하세요.

```python
x = np.zeros((8, 4, 2))
y = x.reshape(-1, 4, 2)
z = y.reshape(8, 2, 4)
```

실제로 배열을 실행해 확인한 것인지, shape를 정적으로 계산한 것인지도 구분하세요.

### 7. pandas → NumPy → 시계열 입력

다음 흐름을 초보자도 이해할 수 있게 설명하세요.

```python
x = train_csv.drop(["casual", "registered", "count"], axis=1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 4, 2)
model.add(LSTM(10, input_shape=(4, 2)))
```

각 코드의 입력·출력 자료 형태, 열 삭제 이유, `fit_transform`과 `transform`의 차이, 열 순서를 보존해야 하는 이유를 포함하세요.

### 8. 분류 코드 정적 검토

`keras2/keras62_2_imdb.py` 또는 `keras2/keras63_sparse1_minist.py`를 기준으로 다음 중 두 가지 이상을 실제 소스에서 찾아 설명하세요.

- label 표현과 loss/activation의 정합성
- `Embedding.input_dim`이 token ID 범위를 포함하는지
- `evaluate`에 전달하는 X와 y의 첫 축 길이가 같은지
- 예측 배열 shape 주석이 실제 예측 대상과 맞는지

실행하지 않은 학습 성능은 만들지 말고, 정적 확인과 실행 확인 불가 영역을 나누어 적으세요.

### 9. Reshape의 원소 수 보존

`(batch, 23, 23, 16)`을 `(batch, 529, 16)`으로 바꿀 수 있는 이유를 총 원소 수 관점에서 설명하세요. 반대로 `(batch, 500, 16)`으로 바꾸려면 어떤 문제가 생기는지 설명하세요. `Reshape`가 데이터의 의미를 자동으로 timestep에 맞게 해석해 주는지도 답하세요.

### 10. 평가 설계와 데이터 누수

train에서 scaler를 `fit`하고 test에서 다시 `fit`하지 않는 이유를 설명하세요. 또한 시계열 데이터를 무작위로 섞어 train/test로 나눌 때 생길 수 있는 문제를, 시간 순서·미래 정보·일반화 평가의 관점에서 설명하세요. 코드에 실제로 발생했다고 단정하지 말고 원칙과 확인 범위를 구분하세요.

## 답변 방법

1~10번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·코드·검증 범위를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
