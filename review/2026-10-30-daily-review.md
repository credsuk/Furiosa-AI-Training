# 2026-10-30 AI 평일 복습 문제

대상 범위: 2026-10-30 학습 코드 없음 → 10-26(`Conv2D`로 이미지 분류)·10-29(`Conv1D`를 표 데이터에)와 이어서, **이미지에 `Conv1D`를 넣은** `keras2/keras68_Conv1D_10_digits.py`(8×8 그림 → 64칸), `keras2/keras68_Conv1D_12_fashion.py`(28×28 → 줄 28개 × 칸 28개), `keras2/keras68_Conv1D_14_cifar100.py`(32×32×3 → 1024×3), `keras2/keras68_Conv1D_08_wine.py`(조기 종료 `monitor` 오타)와 NumPy `reshape` 기초
비고: 10-23(트리 평가), 10-24(도구 호출), 10-25(토크나이저), 10-26(Conv2D 출력 크기·sparse 손실), 10-27(시계열 창·4차원 reshape), 10-28(Conv1D 길이·파라미터·GAP), 10-29(표 데이터 Conv1D·열 순서·스케일 범위)와 겹치지 않게 **"이미지의 세로·가로·색 중 무엇을 시점으로 삼는가"** 를 골랐습니다. 문제에 나오는 모든 숫자는 이 환경(tensorflow-cpu 2.15.1, keras 2.15.0, scikit-learn 1.4.2, pandas 2.1.4, NumPy 1.26.4)에서 실제로 실행해 확인한 값입니다.
문제 수: 7문제 (객관식 4문제, 주관식 3문제)

## 객관식

### 1. `keras2/keras68_Conv1D_10_digits.py`입니다. 손글씨 숫자 데이터는 8×8 그림을 편 것이라 `x_train`이 `(1257, 64)`이고, 코드에서 `(1257, 64, 1)`로 바꿔 넣었습니다. 첫 번째 층을 지난 뒤의 모양은 무엇일까요? (`padding`은 적지 않았습니다.)

```python
model.add(Conv1D(20, (4,), input_shape=(64, 1)))
```

- A. `(1257, 64, 20)`
- B. `(1257, 61, 20)`
- C. `(1257, 60, 20)`
- D. `(1257, 61, 1)`

### 2. `keras2/keras68_Conv1D_12_fashion.py`의 첫 번째 층입니다. 옷 사진 `(60000, 28, 28)`을 그대로 넣었습니다. 이 층의 **파라미터 수**는 몇 개일까요?

```python
model.add(Conv1D(32, (2,), input_shape=(28, 28)))
```

- A. 96개
- B. 1,824개
- C. 1,856개
- D. 1,792개

### 3. NumPy 기초입니다. 32×32 그림을 `reshape(-1)`로 한 줄로 폈을 때, 그림의 **(세로 5번째 줄, 가로 3번째 칸)** 값은 몇 번째 자리에 들어갈까요? (맨 앞을 0번째로 셉니다.)

```python
img = np.arange(32*32).reshape(32, 32)
flat = img.reshape(-1)
```

- A. 143번째
- B. 163번째
- C. 53번째
- D. 96번째

### 4. `keras2/keras68_Conv1D_08_wine.py`의 조기 종료 설정입니다. 이 코드를 그대로 실행하면 **무슨 일이 일어날까요?**

```python
es = EarlyStopping(
    monitor='var_loss',      # val_loss 가 아니다
    mode='min',
    patience=10,
    restore_best_weights=True,
)
model.fit(..., epochs=300, batch_size=32, validation_split=0.2, callbacks=[es])
```

- A. 케라스가 `val_loss`로 이름을 자동으로 고쳐서 원래 의도대로 동작한다
- B. 에폭마다 경고만 출력하고 조기 종료가 한 번도 발동하지 않아 `epochs=300`을 끝까지 돈다
- C. 첫 에폭에서 `ValueError`가 나서 학습이 바로 중단된다
- D. `loss`를 감시하게 되어 훈련 손실이 커지면 멈춘다

## 주관식

### 5. `keras2/keras68_Conv1D_10_digits.py`를 읽고 답해 보세요.

```python
x_train = x_train.reshape(-1, 64, 1)      # (1257, 64) -> (1257, 64, 1)

model = Sequential()
model.add(Conv1D(20, (4,), input_shape=(64, 1)))
model.add(Conv1D(40, (2,)))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

1. 층을 하나씩 지나면서 **출력 모양**이 어떻게 바뀌는지 표로 적어 보세요.
2. 각 층의 **파라미터 수**를 식과 함께 계산하고 총합을 구해 보세요.
3. 이 데이터는 8×8 그림을 64칸으로 편 것입니다. 커널 `(4,)`가 "한 줄 안의 이웃 4칸"만 보는지, 아니면 줄을 넘나드는지 확인해 보세요. 근거도 함께 적어 보세요.
4. 같은 데이터·같은 설정으로 마지막 부분만 바꿔 40에폭씩 학습했더니 다음과 같았습니다. ①과 ②의 정확도가 이렇게 크게 벌어진 이유를 "평균"이라는 말을 넣어 설명해 보세요.

| 마지막 부분 | 테스트 정확도 | 총 파라미터 |
|---|---|---|
| ① `GAP → Dense(64) → Dense(32) → Dense(10)` | 0.4333 | 6,774 |
| ② `Flatten → Dense(64) → Dense(32) → Dense(10)` | 0.9389 | 157,814 |

### 6. `keras2/keras68_Conv1D_12_fashion.py`를 코드 리뷰하듯 읽고 답해 보세요.

```python
x_train = (x_train - 127.5) / 127.5
x_train = x_train.reshape(-1, 28, 28)   # 주석: "4차원 데이터로 변경"

model.add(Conv1D(32, (2,), input_shape=(28, 28)))
model.add(Dropout(0.2))
...
model.add(Flatten())
model.add(Dense(44, activation='relu'))
model.add(Dense(10, activation='softmax'))

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test  = ohe.fit_transform(y_test.reshape(-1, 1))
```

1. `input_shape=(28, 28)`은 옷 사진에서 무엇이 "시점"이고 무엇이 "특징"인가요? 커널 `(2,)`는 사진의 어느 부분을 한 번에 보나요?
2. 이 모델의 **총 파라미터 수**가 66,399개입니다. 그중 가장 큰 층은 어디이고, 파라미터가 몇 개인지 계산해 보세요. (힌트: 마지막 합성곱 출력은 `(22, 41)`입니다.)
3. `y_test`에 `fit_transform`을 쓴 것이 왜 위험한가요? 테스트 데이터에 클래스 하나가 빠졌을 때 실제로 어떤 모양이 되어 어떤 오류가 나는지, 그리고 올바른 코드를 적어 보세요.
4. 주석과 import에서 실제 코드와 맞지 않는 곳을 두 가지 찾아 보세요.

### 7. `keras2/keras68_Conv1D_14_cifar100.py`를 읽고 답해 보세요.

```python
(x_train, y_train), (x_test, y_test) = cifar100.load_data()   # (50000, 32, 32, 3)
x_train = (x_train - 127.5) / 127.5
x_train = x_train.reshape(-1, 1024, 3)
x_test  = x_test.reshape(-1, 1024, 3)

model.add(Conv1D(16, (2,), input_shape=(1024, 3)))
model.add(GlobalAveragePooling1D())
model.add(Dense(41, activation='relu'))
model.add(Dense(100, activation='softmax'))
```

1. `(50000, 32, 32, 3)`을 `(50000, 1024, 3)`으로 바꿨습니다. 여기서 "시점 1개"는 사진의 무엇인가요? 또 시점 0~31과 시점 32~63은 각각 사진의 어느 줄인가요?
2. 커널 `(2,)`가 시점 31과 32를 이웃으로 봅니다. 사진에서는 이 두 점이 어디에 있고, 왜 문제가 될까요? 이런 일이 몇 군데에서 생기는지도 계산해 보세요.
3. 이 모델의 총 파라미터 수와, 첫 합성곱 층의 파라미터 수를 계산해 보세요. 100종류를 맞히는 모델로서 이 숫자가 충분한지도 생각해 보세요.
4. 같은 문제를 `Conv2D(16, (2, 2), input_shape=(32, 32, 3))`으로 시작하면 파라미터가 몇 개일까요? 두 방식의 차이를 "무엇을 이웃으로 보는가"라는 말로 설명해 보세요.

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요. 계산 문제는 계산 과정을 남겨 주세요.
- 1번은 `Conv1D` 출력 길이(`길이-커널+1`), 2번은 `Conv1D` 파라미터 공식, 3번은 NumPy `reshape`의 행 우선 순서, 4번은 `EarlyStopping`의 `monitor` 이름을 묻습니다.
- 5번은 층별 모양·파라미터와 "평균을 내면 위치 정보가 사라진다"는 성질, 6번은 `Conv1D`에 이미지를 넣는 모양과 원-핫 인코더의 `fit`/`transform` 구분, 7번은 줄 경계에서 생기는 이웃 문제와 `Conv1D`·`Conv2D`의 차이를 묻습니다.
- 전체 학습(60,000장, 수백 에폭)은 이 CPU 환경에서 다시 돌리지 않았습니다. 주석에 적힌 정확도와 훈련 시간은 원본 실행 기록이며, 문제의 숫자는 모델 구조 실행과 3~40에폭 소규모 재현으로 확인한 값입니다.
