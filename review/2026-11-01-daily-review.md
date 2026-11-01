# 2026-11-01 AI 주말 복습 문제

대상 범위: 2026-11-01 학습 코드 없음 → 10-26(`Conv2D` 출력 크기·sparse 손실)·10-27(시계열 창)·10-28~10-30(`Conv1D` 계열)·10-31(주가 다중 입력)과 겹치지 않게, 10-14에 **읽기만 하고 넘어간** 전이 학습 묶음을 **실행해서 숫자를 센 결과**로 골랐습니다. 대상은 `keras2/keras72_1_vgg16_include_top.py`, `keras73_vgg16_13_cifar10.py`, `keras74_1_layers.py`~`keras74_4_vgg16_부분동결.py`, `keras75_ALL_TransferLearning.py`, `keras76_all_T_1_cifar10.py`·`3_man_hores.py`입니다.
문제 수: 10문제 (객관식 5문제, 주관식 5문제) — 주말 복습입니다.
비고: 문제에 나오는 모든 숫자·오류 메시지는 이 환경(tensorflow-cpu 2.15.1, keras 2.15.0, NumPy 1.26.4, pandas 2.1.4, CPU 전용)에서 실제로 실행해 확인한 값입니다. 단, 원본 `cifar10` 내려받기가 너무 느려 `keras73` 주석의 정확도(0.7769/0.1)는 재현하지 못했고, `keras76_all_T_3`은 데이터 파일이 없어 오류만 합성 데이터로 재현했습니다.
중복 주의: 2026-10-14 문서·복습에서 `include_top`의 뜻, 가중치 동결, 마지막 Dense 개수 같은 **개념**은 이미 다뤘습니다. 오늘은 같은 파일을 **실측 숫자·동결 전파·재컴파일·층 인덱스** 관점으로 다시 물었습니다.

## 객관식

### 1. 아래 코드에서 `len(model.weights)`는 몇 개일까요? 그리고 그 숫자는 어디서 나왔을까요?

```python
from tensorflow.keras.applications import VGG16
model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
print(len(model.weights))
```

- A. 26개 — 가중치를 가진 합성곱 층이 13개이고, 층마다 커널과 편향 2개씩이기 때문이다
- B. 30개 — 합성곱 13층 × 2 + 입력층 2 + Flatten 2 이기 때문이다
- C. 32개 — VGG16은 가중치를 가진 층이 16개이고 층마다 2개씩이기 때문이다
- D. 16개 — 가중치를 가진 층의 개수를 그대로 센 값이기 때문이다

### 2. VGG16을 32×32 사진에 그대로 쓰려고 아래처럼 적었습니다. 실행하면 어떻게 될까요?

```python
VGG16(weights='imagenet', include_top=True, input_shape=(32, 32, 3))
```

- A. 입력을 자동으로 224×224로 늘려서 `(None, 1000)` 모델이 만들어진다
- B. `include_top=True`가 무시되고 `(None, 1, 1, 512)` 특징 추출기만 만들어진다
- C. `ValueError: ... include_top=True ... input_shape should be (224, 224, 3). Received: input_shape=(32, 32, 3)` 오류가 난다
- D. 모델은 만들어지지만 정확도가 1/1000로 떨어진다

### 3. `keras2/keras74_1_layers.py`에는 이런 주석이 있습니다. 이 주석에 대한 판단으로 **가장 알맞은 것**은 무엇일까요?

```python
# (앞에서 vgg16.trainable = False, include_top=False, 입력 32×32)
model = Sequential([vgg16, Flatten(), Dense(100), Dense(10, activation='softmax')])
print(len(model.weights)) # 30 = 16(vgg16) * 2 (w, b) + 2 (dense) + b
```

- A. 주석이 맞다. VGG16의 층 16개 × 2 = 32에 Dense 2개를 더해 34인데, Flatten을 빼서 30이 된다
- B. 결과 30은 맞지만 식이 틀렸다. `include_top=False`로 Dense 3개를 잘랐으므로 VGG16 몫은 13×2 = 26이고, 26 + 2 + 2 = 30이다
- C. 결과가 틀렸다. 실제로는 32개가 나온다
- D. 결과가 틀렸다. 실제로는 `trainable=False`라서 가중치가 아예 세어지지 않고 2개만 나온다

### 4. 아래 순서로 코드를 실행했습니다. 마지막 줄 뒤에 **무슨 일이 일어날까요?**

```python
v = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
v.trainable = False                       # ① VGG16을 잠근다
seq = Sequential([v, Flatten(), Dense(10, activation='softmax')])
print(len(seq.trainable_weights))         # ② 2  (Dense 커널, 편향)
seq.trainable = True                      # ③
```

- A. VGG16은 ①에서 잠갔으니 그대로이고, 새 Dense만 학습된다 (`2` 유지)
- B. VGG16까지 다시 학습 대상이 되어 `28`이 된다 (Dense 2 + VGG16 26)
- C. 오류가 난다. 안쪽 모델의 `trainable`은 바꿀 수 없다
- D. `0`이 된다. 상위 모델을 켜면 안쪽은 모두 꺼진다

### 5. `keras2/keras76_all_T_3_man_hores.py`는 말/사람 이진 분류입니다. 이 조합으로 `fit`을 시작하면 어떻게 될까요? (정답 라벨은 0과 1)

```python
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

- A. 케라스가 자동으로 `binary_crossentropy`로 바꿔서 정상 학습된다
- B. 손실이 0에 가까워져서 아주 잘 학습된다
- C. `InvalidArgumentError: Received a label value of 1 which is outside the valid range of [0, 1).` 오류로 학습이 시작되지 않는다
- D. 학습은 되지만 정확도가 정확히 0.5에서 멈춘다

## 주관식

### 6. `keras2/keras74_3_layers_부분동결.py`의 일부입니다. 세 가지에 답해 보세요.

```python
model.add(vgg16)                            # layers[0]
model.add(Flatten())                        # layers[?]
model.add(Dense(100))                       # layers[?]
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, trainable=False))
model.add(Dense(10, activation='softmax'))

model.layers[2].trainable = False   # 주석: "Flatten도 쳐준다"
```

1. `model.layers[2]`가 실제로 가리키는 층의 **이름과 종류**는 무엇인가요? Flatten은 몇 번인가요?
2. 이 줄을 실행한 뒤 `len(model.trainable_weights)`는 몇 개이고, 그 안에 들어 있는 가중치 이름 3쌍은 무엇인가요?
3. 파일 아래 주석의 표는 `dense`와 `dense_3`를 `True`로 적고 있습니다. 실제 값은 무엇이며, 이런 실수를 막으려면 층을 번호로 집을 때 무엇을 함께 확인해야 할까요?

### 7. 32×32 사진을 VGG16(`include_top=False`)에 넣으면 출력 모양이 `(None, 1, 1, 512)`입니다.

1. 왜 가로·세로가 1칸만 남는지, VGG16의 풀링 횟수와 함께 설명해 보세요.
2. 이 출력을 `Flatten()`으로 편 뒤 `Dense(100)`을 붙일 때와, `GlobalAveragePooling2D()` 뒤에 `Dense(100)`을 붙일 때의 **파라미터 수**를 각각 구해 보세요. (32×32에서는 둘이 같습니다. 왜 같은지도 설명해 보세요.)
3. 같은 VGG16에 224×224 사진을 넣고 `Flatten()` + `Dense(100)`을 쓰면 파라미터가 몇 개가 될까요? 32×32일 때와 몇 배 차이인지도 계산해 보세요.

### 8. 어떤 모델의 첫 Dense 층을 잠갔다가 다시 풀었습니다. 그런데 학습을 시켜도 그 층의 가중치가 **전혀 바뀌지 않았습니다**(최대 변화 0.0). 이유와 해결 방법을 설명해 보세요.

```python
d1.trainable = False
model.compile(optimizer=SGD(0.1), loss='sparse_categorical_crossentropy')
d1.trainable = True          # 여기서 풀었다고 생각
model.fit(x, y, epochs=3)    # 그런데 d1의 커널 변화는 0.0
```

(참고: 같은 실험에서 `compile()`을 한 번 더 실행한 경우에는 커널 최대 변화가 0.0879였습니다. 초등학생에게 설명하듯 "왜"를 풀어 써 보세요.)

### 9. `keras2/keras75_ALL_TransferLearning.py`를 실행하면 Xception이 `전체 가중치 236 / 학습가능 가중치 156`으로 나옵니다. 그리고 코드 마지막에 "기본적으로 동결되어 있는 가중치가 많다"고 적혀 있습니다.

1. 236 − 156 = 80개의 정체는 무엇인가요? (층 종류와 가중치 이름을 포함해 설명)
2. 이 80개를 "동결된 가중치"라고 부르는 것이 왜 정확하지 않은가요?
3. 같은 파일은 `VGG16()`, `ResNet50()`처럼 **인자 없이** 모델을 만듭니다. 그래서 `include_top=False`로 만든 실측값(ResNet50 318개)과 2개가 다릅니다. 그 2개의 정체는 무엇인가요?

### 10. 기초 — pandas와 NumPy로 층 목록을 세어 봅니다. 아래 코드의 출력 세 줄을 순서대로 적어 보세요.

```python
import pandas as pd, numpy as np

layers = [
    ('Functional', 'vgg16',   False),
    ('Flatten',    'flatten', True),
    ('Dense',      'dense',   False),
    ('Dense',      'dense_1', True),
]
df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

print(len(df[df['Layer Trainable'] == False]))
print(df.loc[df['Layer Type'] == 'Dense', 'Layer Name'].tolist())

shapes = [(512, 100), (100,), (10, 512), (10,)]
print([int(np.prod(s)) for s in shapes], sum(int(np.prod(s)) for s in shapes))
```

## 답변 방법

- 객관식은 `1. A` 처럼 번호와 보기만 적어 주세요.
- 주관식은 번호를 맞춰 자유롭게 적어 주세요. 계산 문제는 식을 함께 적으면 어디서 틀렸는지 짚어 드릴 수 있습니다.
- 모르는 문제는 비워 두셔도 됩니다. 틀린 문제는 다음 복습에서 다른 형태로 다시 나옵니다.
