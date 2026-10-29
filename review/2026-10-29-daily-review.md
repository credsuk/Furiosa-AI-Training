# 2026-10-29 AI 평일 복습 문제

대상 범위: 2026-10-29 학습 코드 없음 → 어제(10-28) `Conv1D`를 "시계열"에 쓴 것과 달리, 표(tabular) 데이터에 `Conv1D`를 얹은 `keras2/keras68_Conv1D_02_diabetes.py`(10개 특징 → `(10,1)`), `keras2/keras68_Conv1D_04_dacon_ddarung.py`(9개 특징 → `(3,3)`), `keras2/keras68_Conv1D_06_cancer.py`(30개 특징 → `(10,3)`)와 pandas·NumPy 기초
비고: 10-23(트리 평가), 10-24(도구 호출), 10-25(토크나이저), 10-26(Conv2D 출력 크기·sparse 손실), 10-27(시계열 창·4차원 reshape), 10-28(Conv1D 길이·파라미터·GAP)과 겹치지 않게 **"표 데이터를 Conv1D에 넣을 때 생기는 일"**(열 순서, 스케일, 제출 파일 누락)을 골랐습니다. 문제에 나오는 모든 숫자는 이 환경(tensorflow-cpu 2.15.1, scikit-learn 1.4.2)에서 실제로 실행해 확인한 값입니다.
문제 수: 7문제 (객관식 4문제, 주관식 3문제)

## 객관식

### 1. `keras2/keras68_Conv1D_06_cancer.py`의 첫 번째 합성곱 층입니다. 유방암 데이터는 특징이 30개라서 `x_train`을 `(398, 10, 3)`으로 바꿔 넣었습니다. 이 층을 지난 뒤의 모양은 무엇일까요? (`padding`은 적지 않았습니다.)

```python
model.add(Conv1D(36, (2,), input_shape=(10, 3)))
```

- A. `(398, 10, 36)`
- B. `(398, 9, 36)`
- C. `(398, 8, 36)`
- D. `(398, 9, 3)`

### 2. 같은 파일의 두 번째 합성곱 층입니다. 바로 앞 층의 출력이 `(None, 9, 36)`일 때 이 층의 **파라미터 수**는 몇 개일까요?

```python
model.add(Conv1D(48, (2,)))
```

- A. 3,456개
- B. 7,008개
- C. 3,504개
- D. 96개

### 3. `keras2/keras68_Conv1D_04_dacon_ddarung.py`는 따릉이 데이터의 특징 9개를 `x_train.reshape(-1, 3, 3)`으로 바꿔 `Conv1D`에 넣습니다. 이 `reshape`에 대한 설명으로 **가장 알맞은 것**은 무엇일까요?

- A. 9개 열 중 값이 큰 열 3개만 골라 시점 3개로 만든다
- B. 9개 열을 3개씩 묶기는 하지만 합성곱은 묶음 경계를 넘지 않으므로, 열 순서를 바꿔도 결과가 똑같다
- C. 열 이름에 `hour`가 들어간 열만 자동으로 찾아 시점 3개로 만든다
- D. 9개 열을 앞에서부터 3개씩 잘라 "시점 3개 × 특징 3개"로 묶는다. 어떤 열이 어느 시점에 들어가는지는 **열 순서**로 정해지므로, 열 순서를 바꾸면 합성곱이 서로 이웃으로 보는 값도 달라진다

### 4. `keras2/keras68_Conv1D_04_dacon_ddarung.py`의 데이터 준비 부분입니다. `train.csv`는 원래 `1459`행 `10`열이었습니다. 이 코드를 실행한 뒤 `x.shape`는 무엇일까요?

```python
train_csv = pd.read_csv(path + "train.csv", index_col=0)   # (1459, 10)
train_csv = train_csv.dropna()
x = train_csv.drop(['count'], axis=1)
```

- A. `(1459, 9)`
- B. `(1328, 10)`
- C. `(1328, 9)`
- D. `(1328, 3, 3)`

## 주관식

### 5. `keras2/keras68_Conv1D_02_diabetes.py`를 읽고 답해 보세요.

```python
x_train = x_train.reshape(-1, 10, 1)      # (331, 10) -> (331, 10, 1)

model = Sequential()
model.add(Conv1D(32, (2,), input_shape=(10,1)))
model.add(Conv1D(64, (2,)))
model.add(Flatten())
model.add(Dense(23))
model.add(Dense(14))
model.add(Dense(1))
```

1. 층을 하나씩 지나면서 **출력 모양**이 어떻게 바뀌는지 표로 적어 보세요. (합성곱 2개, `Flatten`, `Dense` 3개)
2. 각 층의 **파라미터 수**를 식과 함께 계산하고, 총합을 구해 보세요.
3. 이 모델에는 `activation=`이 **하나도 없습니다**(`sigmoid`도 `relu`도 없음). 이런 신경망은 아무리 층을 많이 쌓아도 결국 무엇과 같은 계산이 될까요? 그리고 그렇게 생각한 근거를 한 가지 적어 보세요.
4. `reshape(-1, 10, 1)`은 무슨 뜻인가요? 또 `x_test`는 왜 `111`개일까요? (`train_test_split`에 `test_size`를 적지 않았습니다.)

### 6. 스케일과 `z_evaluate.py`에 대해 답해 보세요.

```python
scaler = MinMaxScaler()
scaler.fit(x_train)             # 훈련 데이터로만 기준을 잡는다
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)
```

1. `fit`을 `x_train`에만 한 이유는 무엇인가요? 이렇게 하면 `x_test` 값이 `0~1` 밖으로 나갈 수 있는데, 실제로 이 파일에서 가장 작은 값과 가장 큰 값은 얼마였나요?
2. 다이아베테스 구조에서 `Flatten()` 대신 `GlobalAveragePooling1D()`를 쓰면 총 파라미터가 `16,406`개에서 몇 개로 줄어들까요? 왜 그렇게 줄어드는지도 설명해 보세요.
3. `keras2/keras68_Conv1D_06_cancer.py` 맨 아래 주석에는 `loss`·`acc` 말고도 `r2 : 0.4597`, `mse : 0.1265`가 찍혀 있습니다. 이 두 값은 어디서 나온 것인지, 그리고 **분류 문제**에서 이 값을 그대로 믿으면 안 되는 이유를 적어 보세요. 힌트: 정답을 0/1로 반올림한 예측으로 `mse`를 구하면 `1 - 정확도`와 어떤 관계가 될까요?

### 7. `keras2/keras68_Conv1D_04_dacon_ddarung.py`를 코드 리뷰하듯 읽고 답해 보세요.

```python
model.add(Conv1D(16, (3,), input_shape=(3,3)))          # (3,3) -> (1,16)
model.add(Conv1D(32, (2,), padding='same'))            # (1,16) -> (1,32)
model.add(Conv1D(62, (2,), padding='same'))            # (1,32) -> (1,62)
model.add(GlobalAveragePooling1D())                    # (1,62) -> (62,)
```

1. 이 파일은 `test.csv`와 `submission.csv`를 읽어 오지만, **제출 파일을 만들 수 없습니다.** 코드에서 무엇이 빠졌는지 두 가지 이상 적어 보세요.
2. 첫 합성곱의 커널이 `(3,)`인데 입력 시점도 `3`개입니다. 이때 `GlobalAveragePooling1D`는 `Flatten`과 결과가 같을까요, 다를까요? 그 이유도 적어 보세요.
3. `root_mean_squared_error`를 쓰려면 scikit-learn 버전이 어때야 하나요? 이 환경에서 확인한 버전도 함께 적어 보세요.
4. `epochs=50000`, `patience=20`으로 설정했습니다. 주석의 훈련 시간은 `8.81초`였습니다. 왜 5만 번을 다 돌지 않았을까요?

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요. 계산 문제는 계산 과정을 남겨 주세요.
- 1번은 `Conv1D` 출력 길이, 2번은 `Conv1D` 파라미터 공식, 3번은 표 데이터를 시간처럼 다룰 때의 의미, 4번은 pandas `dropna`/`drop` 기초를 묻습니다.
- 5번은 층별 모양·파라미터와 "활성화 함수가 없으면 선형"이라는 성질, 6번은 스케일 기준을 훈련 데이터로 잡는 이유와 `z_evaluate.py`가 찍는 회귀용 지표의 함정, 7번은 제출 코드 누락과 `GAP`/`Flatten` 동일 조건, 라이브러리 버전, 조기 종료를 묻습니다.
- 전체 학습(`epochs=10000`, `epochs=50000`)은 이 CPU 환경에서 다시 돌리지 않았습니다. 주석에 적힌 `r2`, `acc`, 훈련 시간은 원본 실행 기록이며, 문제의 숫자는 모델 구조 실행과 30~60에폭 소규모 재현으로 확인한 값입니다.
