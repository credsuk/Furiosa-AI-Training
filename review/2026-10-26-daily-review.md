# 2026-10-26 AI 평일 복습 문제

대상 범위: 2026-10-26 학습 코드 없음 → 최근에 다루지 않은 `keras2/keras63_sparse1_minist.py`·`keras63_sparse3_cifar10.py`(sparse 분류·Conv2D 출력 크기), `keras2/keras64_LSTM01_calfifornia.py`·`keras64_LSTM11_minist.py`(GRU/LSTM 응용·회귀 지표), `keras2/keras65_Reshape1.py`·`Reshape2.py`(Reshape와 LSTM 연결)와 Python·NumPy 기초
비고: 2026-10-26(월)에는 새 학습 코드가 없어, 10-23(트리 모델·병합), 10-24(도구 호출·메모리), 10-25(텍스트 전처리)와 겹치지 않는 **이미지 분류의 shape·손실·Reshape**를 골랐습니다. 문제에 나오는 모든 숫자(shape·파라미터 수·손실·정확도)는 실제 실행으로 확인한 값입니다.
문제 수: 7문제 (객관식 4문제, 주관식 3문제)

## 객관식

### 1. `keras2/keras63_sparse1_minist.py`는 아래처럼 층을 쌓습니다. `padding`을 쓰지 않았을 때 `Conv2D(16, (2,2))`의 출력 모양은 무엇일까요?

```python
model.add(Conv2D(16, (2,2), activation='relu'))  # 입력은 (21, 21, 16)이라고 하자
```

- A. `(20, 20, 16)`
- B. `(19, 19, 16)`
- C. `(21, 21, 32)`
- D. `(20, 20, 8)`

### 2. `keras2/keras63_sparse1_minist.py`의 일부입니다. `y_train`은 `(60000,)` 모양이고 값은 0~9 정수입니다. 이 코드에 대한 설명으로 **맞는 것**은 무엇일까요?

```python
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, ...)
```

- A. `loss='categorical_crossentropy'`로 바꿔도 정수 라벨을 그대로 쓸 수 있다
- B. sparse 손실은 정수 라벨을 그대로 받을 수 있고, `categorical_crossentropy`를 쓰려면 `to_categorical`로 `(60000, 10)` 모양을 만들어야 한다
- C. sparse 손실을 쓰면 마지막 `Dense`의 유닛을 1로 줄여야 한다
- D. sparse 손실을 쓰면 `y_train`을 반드시 `(60000, 10, 1)` 3차원으로 바꿔야 한다

### 3. `keras2/keras65_Reshape1.py`에서 아래 두 줄이 오류 없이 이어집니다. 그 이유로 **가장 알맞은 것**은 무엇일까요?

```python
model.add(Dense(280, input_shape=(28,28)))       # 출력 (N, 28, 280)
model.add(Reshape(target_shape=(28, 28, 10)))    # 출력 (N, 28, 28, 10)
```

- A. `Dense(280)`이 클래스 10개의 점수를 만들어 주기 때문이다
- B. 한 장(샘플)의 숫자 개수가 왼쪽 `28 × 280 = 7,840`, 오른쪽 `28 × 28 × 10 = 7,840`으로 같기 때문이다
- C. `Reshape`가 모자란 숫자를 0으로 채워 주기 때문이다
- D. `Conv2D`가 없으면 Keras가 자동으로 채널을 만들어 주기 때문이다

### 4. `keras2/keras64_LSTM11_minist.py`는 28×28 그림을 `reshape(-1, 784, 1)`로 펴서 `LSTM(10)`에 넣습니다. 주석에 적힌 결과는 `loss 2.3011 / acc 0.1135`입니다. 이 결과가 **거의 찍기 수준**인 까닭으로 가장 알맞은 것은 무엇일까요?

- A. 학습 데이터가 60000개뿐이라 양이 부족해서
- B. 784개의 시간으로 펴면 위아래 이웃 픽셀 정보가 사라지고, 784단계를 거슬러 기억해야 해서 학습이 매우 어려워져서
- C. 마지막 층에 `softmax`를 써서
- D. 픽셀 값을 -1~1로 줄여서

## 주관식

### 5. `keras2/keras63_sparse1_minist.py`의 예측 부분을 읽고 답해 보세요.

```python
y_predict = model.predict(x_test)        # x_test는 10000장
y_predict = np.argmax(y_predict, axis=1)

print(y_predict.shape, y_test.shape)     # 주석: (10,) (10000,)
acc_score = accuracy_score(y_test, y_predict)
```

1. `model.predict(x_test)`의 결과 모양과, `np.argmax(..., axis=1)`을 거친 뒤의 모양을 각각 적어 보세요.
2. 코드 주석 `# (10,) (10000,)`은 실제와 다릅니다. 올바른 값 두 개를 적고, 왜 그렇게 되는지 설명해 보세요.
3. 손으로 계산해 보세요. 아래 `p`에서 `np.argmax(p, axis=1)`과 `np.argmax(p, axis=0)`의 결과는 각각 무엇인가요? 두 결과가 무엇을 뜻하는지도 함께 설명해 보세요.

```python
p = np.array([[0.1, 0.7, 0.2],
              [0.6, 0.3, 0.1]])
```

4. `accuracy_score(y_test, y_predict)`는 값이 제대로 나오는데, `np.mean(y_test == y_predict)`로 직접 비교하면 엉뚱한 값이 나옵니다. 그 이유를 `(10000,1)`과 `(10000,)`의 모양 차이로 설명해 보세요.

### 6. `keras2/keras64_LSTM01_calfifornia.py`를 읽고 답해 보세요.

```python
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target          # x (20640, 8), y (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=253)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, 8, 1)
model.add(GRU(units=10, input_shape=(8, 1)))
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
```

1. `x_train`과 `x_test`의 모양을 학습 데이터 75%, 테스트 25% 기준으로 적어 보세요. 그리고 `reshape(-1, 8, 1)`이 "8"에게 어떤 의미를 주는지 설명해 보세요.
2. `fit_transform`은 학습 데이터에, `transform`은 테스트 데이터에 쓴 이유를 설명해 보세요.
3. 이 문제는 집값(0.15~5.00 사이의 연속된 숫자)을 맞히는 **회귀**입니다. 그런데 `metrics=['acc']`를 썼습니다. 이 지표가 왜 의미가 없는지, 실제로 어떤 값이 나오는지, 대신 무엇을 보면 좋은지 적어 보세요.
4. `GRU(10)`의 파라미터 수를 계산해 보세요. (`GRU`는 문 3개, `LSTM`은 문 4개이며, 각 문의 식은 `입력 수 × 유닛 + 유닛 × 유닛 + 편향`입니다. 입력은 1개입니다.) 같은 조건에서 `LSTM(10)`이면 몇 개가 될까요?

### 7. `keras2/keras65_Reshape2.py`를 읽고 답해 보세요.

```python
model.add(Reshape(target_shape=(28,28,1), input_shape=(28,28)))
model.add(Conv2D(64, (3,3)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Reshape(target_shape=(23*23, 16)))
model.add(LSTM(10))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

1. 층별 출력 모양을 `(None, ...)` 형태로 차례대로 적어 보세요. (첫 `Reshape`부터 마지막 `Dense`까지)
2. `Conv2D` 앞에 `Reshape`를 둔 이유는 무엇일까요? 입력 데이터 `x_train`이 원래 몇 차원이었는지와 함께 설명해 보세요.
3. `(23, 23, 16)`을 `(529, 16)`으로 바꾸는 이유를 LSTM이 원하는 입력 모양과 연결해 설명해 보세요. `23 × 23`은 얼마인가요?
4. `LSTM(10, return_sequences=True)`로 바꾸면 출력 모양이 어떻게 달라질까요? 그렇게 하면 무엇을 할 수 있나요?
5. `LSTM(10)`의 파라미터 수를 계산해 보세요. (입력 특징 16개, 유닛 10개, LSTM은 문 4개)

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요. 계산 문제는 계산 과정을 남겨 주세요.
- 1번은 `Conv2D` 출력 크기 공식(`입력 - 필터 + 1`), 2번은 정답 모양의 차이, 3번은 `Reshape`의 원소 개수, 4번은 긴 시퀀스의 어려움을 묻습니다.
- 이번 문제는 코드의 흐름과 계산 결과를 묻습니다. `epochs=10000` 같은 전체 학습을 돌린 정확도를 묻는 문제는 없습니다.
- 1·2·3·4·5·6·7번의 shape·파라미터 수·오류 메시지·확률 값은 실제 실행(tensorflow-cpu 2.15.1, keras 2.15.0, numpy 1.26.4, scikit-learn 1.9.1, pandas 3.0.6)으로 확인했습니다.
