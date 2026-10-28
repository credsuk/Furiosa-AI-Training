# 2026-10-28 AI 평일 복습 문제

대상 범위: 2026-10-28 학습 코드 없음 → 어제 결론("시계열에는 `Conv1D`")을 확인하는 `keras2/keras67_Conv1D_1.py`(수동 창 + Conv1D 3층 + GAP), `keras2/keras67_Conv1D_2_scale.py`(스케일이 다른 구간·`reshape(-1,3,1)`), `keras2/keras67_Conv1D_3_yena.py`(Conv2D→Conv1D 교체)와 Python·NumPy 기초
비고: 10-23(트리 평가), 10-24(도구 호출), 10-25(토크나이저), 10-26(Conv2D 출력 크기·sparse 손실·Reshape), 10-27(시계열 창 자르기·4차원 reshape·콜백)과 겹치지 않게 **`Conv1D`의 시간축·파라미터·GAP**를 골랐습니다. 문제에 나오는 모든 숫자는 이 환경(tensorflow-cpu 2.15.1)에서 실제 실행해 확인한 값입니다.
문제 수: 7문제 (객관식 4문제, 주관식 3문제)

## 객관식

### 1. `keras2/keras67_Conv1D_1.py`의 첫 층입니다. 입력이 `(7, 3, 1)`일 때 이 층을 지난 뒤의 모양은 무엇일까요? (`padding`은 아무것도 적지 않았습니다.)

```python
model.add(Conv1D(filters=20, kernel_size=2, input_shape=(3,1)))
```

- A. `(7, 3, 20)`
- B. `(7, 1, 20)`
- C. `(7, 2, 20)`
- D. `(7, 2, 1)`

### 2. `keras2/keras67_Conv1D_3_yena.py`의 두 번째 합성곱 층입니다. 바로 앞 층의 출력이 `(None, 138, 36)`일 때 이 층의 **파라미터 수**는 몇 개일까요?

```python
model.add(Conv1D(48, (4)))
```

- A. 192개
- B. 6,960개
- C. 6,912개
- D. 48개

### 3. 10-27에는 시계열 144개를 12×12로 접어 `Conv2D`에 넣었고, 오늘은 접지 않고 `Conv1D`에 넣었습니다. `Conv1D`가 시계열에 더 자연스러운 이유로 **가장 알맞은 것**은 무엇일까요?

- A. 커널이 시간축을 따라 한 칸씩 움직이며 이웃한 시점끼리만 계산하고, 같은 가중치를 모든 시점에서 다시 쓰므로 시점이 늘어도 파라미터가 늘지 않는다
- B. 12×12로 접으면 격자의 이웃 칸이 항상 1시점 차이가 되므로 `Conv2D`가 더 정확하다
- C. `Conv1D`는 특징(채널)을 모두 펴서 `Dense`처럼 계산하므로 시간 위치 정보가 그대로 보존된다
- D. `Conv1D`는 커널 크기가 입력 길이와 정확히 같아야만 동작하므로 시간 순서가 강제로 지켜진다

### 4. `keras2/keras67_Conv1D_1.py`는 합성곱 뒤에 `GlobalAveragePooling1D()`를 넣었습니다. 이 층이 하는 일과 그 영향으로 **가장 알맞은 것**은 무엇일까요?

```python
model.add(Conv1D(10, 1))              # (7, 1, 10)
model.add(GlobalAveragePooling1D())   # (7, 10)
model.add(Dense(36, activation='relu'))
```

- A. 시점 방향의 값을 **순서대로 이어 붙여** 1차원으로 펴므로 시간 순서가 그대로 보존된다
- B. 시점 방향에서 **가장 큰 값 하나**만 골라 내므로 길이가 1이 된다
- C. 채널 수를 1개로 줄여 마지막 `Dense`에 바로 넘긴다
- D. 시점 방향을 **평균** 내어 채널만 남기므로 다음 `Dense`의 입력 크기가 시점 수와 무관해지고, 대신 "언제" 그런 값이었는지는 사라진다

## 주관식

### 5. `keras2/keras67_Conv1D_1.py`를 읽고 답해 보세요.

```python
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])   # (7, 3)
y = np.array([4,5,6,7,8,9,10])                                            # (7,)
x = x.reshape(x.shape[0], x.shape[1], 1)      # (7, 3, 1)

model = Sequential()
model.add(Conv1D(filters=20, kernel_size=2, input_shape=(3,1)))   # 주석: "3개가 들어가서 3개가 나옴"
model.add(Conv1D(20, 2))
model.add(Conv1D(10, 1))
model.add(GlobalAveragePooling1D())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

x_predict = np.array([8,9,10]).reshape(1, 3, 1)
# 주석: loss : 4.368691861600382e-07 / [8,9,10]의 결과 : [[11.001902]]
```

1. `x.reshape(x.shape[0], x.shape[1], 1)`이 하는 일을 말로 설명해 보세요. 같은 결과를 내는 더 짧은 표현은 무엇인가요?
2. 세 개의 `Conv1D`를 차례로 지날 때 모양이 어떻게 바뀌는지 표로 적어 보세요. (`(7,3,1)`에서 시작)
3. 각 층의 파라미터 수를 식과 함께 계산해 보세요. 합계는 몇 개인가요? (`Dense(36)`의 입력은 GAP를 지난 뒤의 10개입니다)
4. 주석 "3개가 들어가서 3개가 나옴"은 맞는 말일까요? 아니라면 실제로 몇 개가 나오는지 쓰고, 3개를 유지하려면 어떤 옵션을 줘야 하는지 적어 보세요.
5. 첫 `Conv1D`에는 `activation`을 적지 않았습니다. 이때 이 층이 쓰는 활성화 함수는 무엇이며, "곱하고 더하기"만 하는 층이 되는 것과 어떤 관계가 있나요?
6. 이 모델은 학습한 데이터를 그대로 다시 평가해서 loss가 거의 0에 가깝고, `[8,9,10]`의 답도 정답(11)에 아주 가깝게 나옵니다. 이 숫자만 보고 "새로운 데이터도 잘 맞힌다"고 말할 수 있을까요? 그 이유를 적어 보세요.

### 6. `keras2/keras67_Conv1D_2_scale.py`를 읽고 답해 보세요.

```python
x = np.array([[1,2,3], ..., [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])   # (13, 3)
y = np.array([4, ..., 13, 50, 60, 70])               # (13,)
x = x.reshape(x.shape[0], x.shape[1], 1)             # (13, 3, 1)

x_predict = np.array([50,60,70])
x_predict = x_predict.reshape(-1, 3, 1)
# print(x_predict.shape) # (3, 1)

model = Sequential([
    Conv1D(10, 2, input_shape=(3,1)), Conv1D(20, 2), Flatten(),
    Dense(64, activation='relu'), Dense(32, activation='relu'),
    Dense(8, activation='relu'), Dense(1)])
model.compile(loss='mse', optimizer=Adam(learning_rate=0.05))
model.fit(x, y, epochs=50000, batch_size=12, validation_split=0.1, callbacks=[es, rlr])
# 주석: loss : 7.206088776001707e-05 / y_predict 결과 : [[80.020004]]
```

1. `np.array([50,60,70])`의 모양과, `reshape(-1, 3, 1)` 뒤의 모양을 각각 적어 보세요. 주석의 `(3, 1)`은 맞는 말일까요?
2. 만약 정말로 `reshape(3, 1)`로 만들어 모델에 넣으면 어떻게 될까요? "샘플 3개, 시점 1개, 특징 1개"라는 뜻이 왜 모델의 요구와 어긋나는지도 함께 설명해 보세요.
3. 전체 13개, `batch_size=12`, `validation_split=0.1`이면 한 에폭에서 **학습에 쓰는 샘플 수**와 **검증에 쓰는 샘플 수**, 그리고 **한 에폭의 스텝 수**는 각각 몇 개일까요?
4. 같은 구조·같은 설정에서 실제로 돌려 본 결과가 아래와 같습니다. 왜 이렇게 갈렸는지, 그리고 이 표에서 무엇을 배워야 하는지 적어 보세요.

| 설정 | 멈춘 에폭 | loss | `[50,60,70]` 예측 |
|---|---|---|---|
| `validation_split=0.1` (파일과 동일) | 258 | 0.0014 | 79.62 |
| 검증 분할 없음·콜백 없음 | 15000 (끝까지) | 492.55 | 20.30 |

5. `Flatten()`을 지난 뒤의 모양이 `(None, 20)`인 이유를 `Conv1D(20, 2)`의 출력 모양과 연결해 설명해 보세요.
6. 기초 확인: 이 파일의 `x`는 `(13, 3, 1)`입니다. `x.shape[0]`, `x.shape[1]`, `x.shape[2]`는 각각 무엇을 뜻하나요? 또 `len(x)`의 값은 얼마인가요?

### 7. `keras2/keras67_Conv1D_3_yena.py`를 읽고 답해 보세요. (10-27의 `keras66_jena_CNN.py`와 데이터 준비가 같고 모델만 바뀐 파일입니다.)

```python
x_data = datasets[:-288].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)  # (420263, 13)
# ... split_x로 144개씩 자르고 (420120, 144, 13), train_test_split, MinMaxScaler ...

model.add(Conv1D(36, (4), input_shape=(144, 13)))     # 10-27 버전은 Conv2D(36, (4,4)) + 12x12 접기
model.add(Dropout(0.3))
model.add(Conv1D(48, (4)))
model.add(Dropout(0.2))
model.add(Conv1D(24, (2)))
model.add(GlobalAveragePooling1D())
model.add(Dense(108, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(28, activation='relu'))
model.add(Dense(144, activation='relu'))

x_predict = datasets[-288:-144].drop(['wd (deg)'], axis=1)
x_predict = x_predict.to_numpy().reshape(1, 144, 13)
```

1. 10-27의 `Conv2D(36, (4,4))`는 파라미터가 7,524개였습니다. `Conv1D(36, (4))`는 몇 개인지 식과 함께 계산해 보세요. 왜 이렇게 차이가 날까요?
2. 세 개의 `Conv1D`를 지나며 시점 길이가 `144`에서 어떻게 줄어드는지 `141`, `138`, `137`을 직접 계산으로 확인해 보세요.
3. `GlobalAveragePooling1D()` 뒤의 `Dense(144)`에 대한 설명으로 맞는 것을 두 가지 적어 보세요. (힌트: 137개 시점은 어디로 갔나요? 마지막 층의 `relu`는 무엇을 못 하게 하나요?)
4. 훈련은 `T (degC)`를 지웠는데 예측 데이터는 `wd (deg)`를 지웁니다. 이 코드가 **오류 없이 실행되는** 이유와, 그런데도 **결과가 잘못되는** 이유를 각각 적어 보세요. 올바른 한 줄도 써 보세요.
5. `x_predict`는 `MinMaxScaler`로 변환되지 않았습니다. 학습 때 만든 `scaler`를 그대로 쓰려면 어떤 순서로 코드를 써야 할까요?
6. 기초 확인: pandas에서 `.drop(['wd (deg)'], axis=1)`의 `axis=1`은 "행"과 "열" 중 어느 쪽을 지우라는 뜻인가요? 만약 `axis=0`으로 쓰면 어떤 일이 생길까요?

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요. 계산 문제는 계산 과정을 남겨 주세요.
- 1번은 `Conv1D`의 출력 길이, 2번은 `Conv1D`의 파라미터 공식, 3번은 시간축 합성곱의 의미, 4번은 `GlobalAveragePooling1D`의 역할을 묻습니다.
- 5번은 `Conv1D` 3층의 모양·파라미터·주석 검증과 "훈련 데이터로 평가한 점수"의 의미, 6번은 `reshape(-1,3,1)`과 작은 데이터셋의 재현성, 7번은 `Conv2D`→`Conv1D` 교체 효과와 예측 데이터의 열 삭제 실수를 묻습니다.
- 전체 학습을 돌린 정확도를 묻는 문제는 없습니다. 예나 데이터 CSV가 이 환경에 없어 원본 학습 결과(주석의 loss·RMSE)는 재현하지 않았고, 문제의 숫자는 모델 구조 실행과 소규모 재현으로 확인한 값입니다.
