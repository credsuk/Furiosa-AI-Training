# 2026-10-02 AI 학습 일일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-02 (금요일)
- 기준 최신 SHA: `579b2af291656f6a80cd6959d0a489782b8e54b8` (`day22`, 17:47 KST)
- 주제: sparse categorical classification, `pd.get_dummies`, CNN 입력 shape, `Reshape`, LSTM/GRU 시계열 변환
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)

## 복습 핵심

- `sparse_categorical_crossentropy`는 정수 class label을 그대로 사용할 수 있어 one-hot 변환이 필수는 아닙니다.
- `pd.get_dummies`로 one-hot DataFrame을 만들면 categorical crossentropy와 다중 출력 `softmax` 조합을 사용할 수 있습니다.
- 이미지 입력은 CNN에 일반적으로 `(batch, height, width, channel)` 형태를 전달하며, 흑백 MNIST의 channel 축은 1입니다.
- `Reshape`는 원소 수를 보존하면서 축의 모양만 바꾸는 연산입니다. `(-1, 28, 28)`을 `(batch, 28, 28, 1)`로 바꾸려면 샘플별 원소 수가 맞아야 합니다.
- LSTM/GRU에 넣을 시계열은 `(batch, timestep, feature)`입니다. 이미지의 공간 축을 timestep과 feature로 펼칠 때 그 의미가 달라지므로 shape와 데이터 순서를 함께 확인해야 합니다.
- 코드에 적힌 주석·실행 예시는 실제 실행 결과와 구분해야 합니다. day22 소스에는 `Embedding(np.max(x_train), ...)`, `monitor='var_loss'`, `model.evaluate(x_train, y_test)` 및 `y_predict.shape` 주석처럼 실행 전 점검이 필요한 부분이 있습니다.

## 객관식

### 1. sparse categorical loss

다음 코드에서 `y_train`이 `[0, 2, 1, ...]`처럼 정수 class label이고 마지막 층이 `Dense(3, activation='softmax')`일 때 가장 적절한 조합은 무엇인가요?

- **A.** `loss='sparse_categorical_crossentropy'`를 사용하고 `y_train`을 정수 형태로 유지한다.
- **B.** `loss='binary_crossentropy'`를 사용하고 모든 label을 문자열로 바꾼다.
- **C.** `loss='mse'`를 사용하고 마지막 층의 unit을 1개로 줄인다.
- **D.** `loss='sparse_categorical_crossentropy'`를 사용하려면 반드시 `to_categorical`을 먼저 적용한다.

### 2. pandas one-hot과 shape

`y = pd.get_dummies(y)`의 일반적인 결과로 맞는 것은 무엇인가요?

- **A.** 각 class가 열이 된 DataFrame이 되며 행 수는 원래 label 개수와 같다.
- **B.** 각 행을 하나의 실수 scalar로 평균낸 Series가 된다.
- **C.** 입력 이미지의 channel 축이 자동으로 3개가 된다.
- **D.** class 개수와 무관하게 항상 shape가 `(n, 1)`이 된다.

### 3. MNIST CNN 입력

MNIST 이미지 배열의 shape가 `(60000, 28, 28)`일 때 `Conv2D`에 넣기 위한 일반적인 reshape 결과는 무엇인가요?

- **A.** `(60000, 28, 28, 1)`
- **B.** `(28, 28, 60000, 1)`
- **C.** `(60000, 1, 28)`
- **D.** `(60000, 784)`

### 4. Reshape와 LSTM

`keras2/keras65_Reshape2.py`에서 합성곱 출력 `(batch, 23, 23, 16)`을 `Reshape(target_shape=(23*23, 16))` 뒤 `LSTM(10)`에 전달하는 설명으로 맞는 것은 무엇인가요?

- **A.** 공간의 두 축을 하나의 timestep 축으로 합쳐 `(batch, 529, 16)`을 만든다.
- **B.** 모든 픽셀을 삭제해 `(batch, 10)`만 만든다.
- **C.** channel 16개를 timestep 16개로 바꾸고 feature는 529개가 된다.
- **D.** `Reshape`가 LSTM의 hidden unit 수를 자동으로 529개로 바꾼다.

## 주관식

### 5. Python·NumPy·shape 추적

다음 변환의 각 단계 shape를 적고, `-1`이 의미하는 바를 설명하세요.

```python
x = np.zeros((60000, 28, 28))
x = x.reshape(-1, 28, 28, 1)
```

또한 `(batch, height, width, channel)`의 각 축을 설명하세요. 실제 TensorFlow 실행 없이 shape 계산만 한 것인지도 구분해 적으세요.

### 6. 코드 검토: 분류 모델의 label·평가 정합성

`keras2/keras62_2_imdb.py` 또는 `keras2/keras63_sparse1_minist.py`에서 다음 중 **두 가지 이상**의 문제를 찾아 설명하고 수정 방향을 작성하세요.

- binary/multiclass label과 마지막 activation·loss의 정합성
- `Embedding`의 `input_dim`이 token ID의 최대값을 모두 포함하는지
- `model.evaluate()`에 전달하는 X와 y의 샘플 수가 같은지
- `y_predict`와 `y_test`의 shape가 실제 코드와 일치하는지

실제 학습을 실행하지 않았다면 실행 결과를 만들지 말고 정적 검토와 확인 불가 영역을 나누어 쓰세요.

### 7. pandas 전처리와 시계열 재구성

`keras2/keras64_LSTM05_kaggle_bike.py`의 다음 흐름을 설명하세요.

```python
x = train_csv.drop(["casual", "registered", "count"], axis=1)
x_train = scaler.fit_transform(x_train)
x_train = x_train.reshape(-1, 4, 2)
model.add(LSTM(units=10, input_shape=(4, 2)))
```

`drop(..., axis=1)`, `fit_transform`과 `transform`의 역할, `(batch, timestep, feature)`의 의미를 각각 적고, 데이터 열 순서를 임의로 바꾸면 안 되는 이유를 설명하세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·코드·검증 범위를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
