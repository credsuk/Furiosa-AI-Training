# 2026-10-27 AI 평일 복습 문제

대상 범위: 2026-10-27 학습 코드 없음 → 최근에 다루지 않은 `keras2/keras66_jena_CNN.py`(시계열 창 자르기·4차원 reshape·회귀 지표·콜백)와 Python·NumPy·pandas 기초
비고: 10-23(트리 평가), 10-24(도구 호출), 10-25(텍스트 전처리), 10-26(sparse 손실·Reshape)와 겹치지 않는 **시계열을 그림처럼 접어 Conv2D에 넣기**를 골랐습니다. 문제에 나오는 shape·파라미터 수·오류 메시지·파일 이름은 원본 CSV가 없어 **같은 구조의 합성 데이터로 실제 실행해 확인한 값**입니다.
문제 수: 7문제 (객관식 4문제, 주관식 3문제)

## 객관식

### 1. `keras2/keras66_jena_CNN.py`는 앞 144개 시점을 보고 그다음 144개 시점의 기온을 맞히려고 합니다. 아래 코드에서 입력 창과 정답 창의 관계로 **맞는 것**은 무엇일까요?

```python
x_data = datasets[:-288].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
y_data = datasets[144: -144]['T (degC)'].to_numpy(dtype=np.float32)
# 이후 둘 다 split_x(..., 144)로 144개씩 창을 자른다
```

- A. 입력 창 0은 원본 0~143줄, 정답 창 0은 원본 144~287줄이라서 "다음 144개를 맞히는" 문제가 된다
- B. 입력 창 0과 정답 창 0이 같은 0~143줄에서 시작해서 "지금 값을 베끼는" 문제가 된다
- C. 정답 창 0이 입력 창 0보다 144줄 **앞선** 구간이라서 "과거를 맞히는" 문제가 된다
- D. `y_data`를 `datasets[:-288]`로 바꾸어야만 미래를 맞히는 문제가 된다

### 2. 같은 파일에서 `x_data = x_data.reshape(-1, 12, 12, 13)`이 오류 없이 실행됩니다. 그 이유와 의미로 **가장 알맞은 것**은 무엇일까요?

- A. 한 샘플의 원소 개수가 `144 × 13 = 1,872`, 바꾼 뒤에도 `12 × 12 × 13 = 1,872`으로 같기 때문이고, 채널 13개는 그 시점의 특징, `(i, j)` 칸은 시간 `12i + j`를 뜻한다
- B. 144가 12로 나누어떨어지기만 하면 되고, 채널 13개는 13개의 시간을, 12×12 칸은 12×12개의 특징을 뜻한다
- C. 특징 13개를 12개로 줄여서 원소 개수를 맞추기 때문이다
- D. 4차원으로 바꾸는 것은 불가능하며, 이 줄은 실제로는 오류가 난다

### 3. 이 파일은 기온(연속된 숫자)을 맞히는 회귀인데 `metrics=['acc']`를 썼고, 주석에 `acc : 0.0383`이 적혀 있습니다. 이 숫자에 대한 설명으로 **맞는 것**은 무엇일까요?

- A. 소수점까지 정확히 같은 값을 맞힌 비율을 세는 지표라서 거의 항상 0 근처에 머물고, 의미가 없다
- B. 10개 클래스 중 3.83%를 맞혔다는 뜻이라서 성능이 나쁘다는 것을 알려 준다
- C. `mse` 값이 0.0383이라는 뜻이다
- D. `RMSE`가 0.0383도라는 뜻이라서 아주 좋은 모델이다

### 4. 이 모델의 마지막 층은 `model.add(Dense(144, activation='relu'))`입니다. 이 선택의 문제로 **가장 알맞은 것**은 무엇일까요?

- A. 출력층의 `relu`가 음수를 0으로 만들어서 0도 아래 기온을 절대로 낼 수 없다. 회귀의 출력층은 활성화 함수를 쓰지 않는 것이 기본이다
- B. `relu`는 아주 큰 값만 잘라 내므로 기온 예측에는 아무 문제가 없다
- C. `softmax`를 써야 기온을 확률로 표현할 수 있다
- D. 출력층에 `relu`를 쓰면 Keras가 자동으로 선형(linear)으로 바꿔 준다

## 주관식

### 5. `keras2/keras66_jena_CNN.py`의 데이터 준비 부분을 읽고 답해 보세요. (원본 데이터는 420,551줄 × 14열이고, `train_test_split`은 옵션을 주지 않아 테스트가 25%입니다.)

```python
x_data = datasets[:-288].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
y_data = datasets[144: -144]['T (degC)'].to_numpy(dtype=np.float32)
# print(x_data.shape, y_data.shape)   # (420263, 13) (420263,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        aaa.append(dataset[i : (i+size)])
    return np.array(aaa)

x_data = split_x(x_data, 144)
y_data = split_x(y_data, 144)
# print(x_data.shape, y_data.shape)   # (420120, 144, 13) (420120, 144)

x_data = x_data.reshape(-1, 12, 12, 13)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=523, shuffle=False)
```

1. `x_data`가 `(420263, 13)`, `y_data`가 `(420263,)`이 되는 과정을 설명해 보세요. `datasets[:-288]`은 몇 줄이고, `datasets[144:-144]`는 몇 줄인가요?
2. 창의 개수 `420120`이 어떻게 나오는지 식으로 적어 보세요. `range(len(dataset) - size + 1)`에서 `+1`을 빼면 무엇이 달라지나요?
3. `train_test_split` 뒤 `x_train`, `x_test`, `y_train`, `y_test`의 모양을 각각 적어 보세요.
4. `reshape(-1, 12, 12, 13)`에서 `-1`은 무엇을 뜻하나요? 한 샘플의 원소 개수가 양쪽에서 같은지 계산으로 확인해 보세요.
5. 기초 확인: `datasets[-288:-144]`는 몇 줄인가요? 그리고 아래 두 줄의 차이를 설명해 보세요.

```python
aaa = []
aaa.append(x_data[i:(i+144)])
# ... 반복 후
aaa = np.array(aaa)
```

6. 이 4차원 배열 전체는 `float32`로 몇 GiB쯤 될까요? `420120 × 12 × 12 × 13 × 4바이트`로 계산해 보세요. (계산 결과가 2~3GiB 정도라면 맞습니다.)
7. `shuffle=False`를 쓴 이유를 시계열의 성질과 연결해 설명해 보세요.

### 6. `keras2/keras66_jena_CNN.py`의 예측 준비 부분에는 실행하면 바로 문제가 되는 부분이 있습니다. 읽고 답해 보세요.

```python
y_cor = datasets[-144:]['T (degC)']                        # (144,)

x_predict = datasets[-288:-144].drop(['wd (deg)'], axis=1)
x_predict = x_predict.to_numpy()
x_predict = x_predict.reshape(1, 144, 13)

model.add(Conv2D(36, (4, 4), input_shape=(12, 12, 13)))
# ... 학습 후
y_predict = model.predict(x_test)
```

1. 훈련 데이터는 `T (degC)` 열을 지웠는데 `x_predict`는 `wd (deg)` 열을 지웁니다. 이때 두 데이터의 열 13개가 어떻게 달라지는지, 어떤 열이 훈련에만 있고 어떤 열이 예측에만 있는지 적어 보세요.
2. `x_predict`의 모양 `(1, 144, 13)`을 그대로 `model.predict`에 넣으면 어떤 오류가 날까요? 오류 메시지의 핵심 문구를 적어 보세요.
3. 위 코드를 고쳐서 "예측용 데이터 1개를 모델에 넣을 수 있는 4차원 모양"으로 만드는 줄을 쓰세요. (정답 열을 지우고, 12×12로 접는 것까지)
4. `x_predict`는 `MinMaxScaler`로 변환되지 않았습니다. 학습 때 만든 `scaler`를 그대로 쓰려면 어떤 순서로 코드를 써야 할까요? (펴기 → `transform` → 다시 접기)
5. `y_cor`는 만들어만 두고 쓰이지 않습니다. 이 변수를 살려서 "예측값과 실제값의 RMSE"를 출력하는 코드를 한 줄 써 보세요.

### 7. 같은 파일의 컴파일·콜백·평가 부분을 읽고 답해 보세요.

```python
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                      filepath="./_save/keras66/kaggle_jena1_{epoch:04d}-{val_loss:.4f}.keras", verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=50, verbose=1, factor=0.1)
es  = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100000, batch_size=700, validation_split=0.2, callbacks=[es, rlr, mcp])

loss = model.evaluate(x_test, y_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
```

1. `EarlyStopping(patience=100, restore_best_weights=True)`는 각각 무엇을 하나요? `restore_best_weights`가 없으면 무엇이 문제일까요?
2. `ReduceLROnPlateau(factor=0.1, patience=50)`에서 처음 학습률이 `0.001`이었다면, 두 번 줄어든 뒤의 학습률은 얼마인가요? `factor`는 곱하는 값인가요, 나누는 값인가요?
3. `ModelCheckpoint`의 `filepath` 규칙대로라면, 12번째 에포크에서 `val_loss`가 `63.5`였을 때 저장되는 파일 이름은 무엇인가요? 또 `save_best_only=True`라서 어떤 에포크의 모델이 남나요?
4. 주석에 `loss : 76.50410461425781`이 적혀 있습니다. 이 값과 `RMSE`의 관계를 식으로 쓰고, `RMSE`를 계산해 보세요. (루트를 씌우면 8.7 정도입니다.)
5. 회귀에서 `metrics=['acc']` 대신 무엇을 보면 좋을까요? 두 가지 이상 적어 보세요.
6. "늘 학습 데이터의 평균 기온만 답하는 모델"을 베이스라인이라고 합니다. 이 베이스라인의 RMSE가 내 모델 RMSE보다 작게 나왔다면 무엇을 뜻할까요?

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요. 계산 문제는 계산 과정을 남겨 주세요.
- 1번은 창 정렬, 2번은 4차원 reshape의 원소 개수와 의미, 3번은 회귀에서의 `acc`, 4번은 출력층 활성화 함수를 묻습니다.
- 5~7번은 창 개수·분할 모양 계산, 예측 데이터의 열 삭제 실수와 shape 오류, 콜백 3종과 `mse`→`RMSE` 해석을 묻습니다.
- 전체 학습(`epochs=100000`)을 돌린 정확도를 묻는 문제는 없습니다. 원본 CSV가 이 환경에 없어 원본 학습 결과는 재현하지 않았고, 문제의 숫자는 합성 데이터 실행으로 확인한 값입니다.
