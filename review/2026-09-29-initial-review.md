# 전체 범위 최초 복습 자료

작성일: 2026-09-29
기준 저장소: `Furiosa-AI-Training`
기준 브랜치: `main`

## 1. 저장소 학습 흐름

현재 저장소는 작은 Python 연산 예제에서 시작해 NumPy 배열, Keras 회귀 모델, 학습·테스트 분리, 평가 지표, 실제 데이터셋과 CSV 전처리로 확장되는 학습 기록입니다.

전체적인 흐름은 다음과 같습니다.

```text
Python 기본
  → NumPy 배열과 shape
  → 단순 선형 회귀
  → 여러 Dense 층을 가진 신경망
  → batch size와 다중 입력·출력
  → train/test 분리
  → 시각화
  → 실제 데이터셋 회귀
  → MSE·R2·RMSE 평가
  → Pandas CSV 전처리와 제출
```

## 2. Python 기본

`test0001.py`에서는 변수, 덧셈, 곱셈, 출력, 실행 단축키를 확인합니다.

```python
a = 1
b = 2
c = a + b
f = c * 3
print(f)
```

핵심은 변수에 값을 저장하고, 연산 결과를 다시 변수에 저장하며, `print()`로 확인하는 흐름입니다. Python은 들여쓰기가 문법이므로 함수와 조건문을 확장할 때 들여쓰기를 지켜야 합니다.

## 3. NumPy 배열과 shape

`keras07_행열.py`에서 배열의 차원과 크기를 확인합니다.

- `(3,)`: 원소 3개인 1차원 벡터
- `(1, 3)`: 1행 3열 행렬
- `(3, 2)`: 3행 2열 행렬
- `(2, 2, 2)`: 3차원 배열

`shape`는 단순히 원소 수가 아니라 각 축의 크기를 순서대로 표현합니다. Keras에서는 보통 첫 번째 축이 샘플 수이고, 두 번째 축이 특성 수입니다.

예를 들어:

```python
x = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
```

의 shape는 `(5, 2)`입니다. 샘플 5개가 있고, 각 샘플에 특성 2개가 있습니다. 따라서 `input_dim=2`가 됩니다.

`x.T` 또는 `tf.transpose(x)`는 행과 열을 바꿉니다. 다만 transpose는 데이터의 배치 방향을 바꾸는 것이므로, 바꾼 뒤 샘플 수와 특성 수가 의도와 맞는지 확인해야 합니다.

## 4. Keras 회귀 모델 기본 흐름

`keras01.py`부터 대부분의 파일은 다음 4단계를 따릅니다.

```python
# 1. 데이터
x = np.array([...])
y = np.array([...])

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. 컴파일과 학습
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

# 4. 평가와 예측
loss = model.evaluate(x, y)
result = model.predict(np.array([4]))
```

- `Sequential`: 레이어를 순서대로 쌓는 모델
- `Dense`: 이전 레이어의 모든 출력과 연결되는 완전연결층
- `input_dim`: 입력 특성 개수
- `compile`: 손실 함수와 optimizer 설정
- `fit`: 학습 데이터로 가중치 업데이트
- `evaluate`: 주어진 데이터에서 손실 계산
- `predict`: 입력에 대한 예측값 생성

## 5. 손실 함수와 optimizer

저장소의 회귀 예제는 주로 `loss='mse'`를 사용합니다.

MSE는 실제값과 예측값의 차이를 제곱한 뒤 평균낸 값입니다.

```text
MSE = 평균((실제값 - 예측값)²)
```

값이 작을수록 예측 오차가 작다는 뜻입니다. 제곱하기 때문에 큰 오차에 더 민감합니다.

`optimizer='adam'`은 손실을 줄이도록 가중치를 조정하는 최적화 방법입니다. `epochs`는 전체 학습 데이터를 반복해서 보는 횟수이고, `batch_size`는 한 번에 사용해 가중치를 업데이트하는 샘플 묶음의 크기입니다.

## 6. 단순 모델과 깊은 모델

`keras03.py`는 데이터 관계가 단순하지 않을 때 단일 Dense 층만으로는 충분하지 않을 수 있음을 보여줍니다. `keras04_deep1.py`, `keras05_deep2.py`, `keras06_batch.py`는 레이어를 여러 개 쌓고 노드 수와 batch size를 바꿔봅니다.

레이어를 많이 쌓거나 노드를 늘리는 것이 항상 좋은 것은 아닙니다. 모델이 복잡해지면 학습 데이터에 지나치게 맞는 과적합 위험도 커집니다. epoch와 layer 수는 성능·학습 시간·일반화 성능을 함께 보면서 정해야 합니다.

## 7. 다중 입력과 다중 출력

`keras08_*` 파일에서는 입력 특성이 여러 개인 경우를 다룹니다.

- `x.shape == (샘플 수, 특성 수)`
- `input_dim`은 특성 수
- `y.shape == (샘플 수,)`이면 출력 1개
- `y.shape == (샘플 수, 출력 수)`이면 출력 여러 개

예를 들어 `x.shape == (10, 3)`이고 `y.shape == (10, 2)`라면 샘플 10개, 입력 특성 3개, 출력값 2개입니다. 마지막 Dense 레이어의 노드 수는 출력 수와 맞춰야 합니다.

## 8. 학습 데이터와 테스트 데이터

`keras09_train_test*.py`는 같은 데이터를 모두 학습시키지 않고 일부를 평가용으로 분리하는 과정을 보여줍니다.

- train data: 모델의 가중치를 학습하는 데이터
- test data: 학습에 사용하지 않고 일반화 성능을 확인하는 데이터

`train_test_split(x, y, random_state=...)`는 X와 Y의 대응 관계를 유지하면서 데이터를 나눕니다. X와 Y를 따로 섞으면 입력과 정답의 짝이 깨질 수 있으므로 함께 넘겨야 합니다.

`random_state`는 난수 분할을 재현하기 위한 seed입니다. 같은 seed를 사용하면 같은 분할 결과를 다시 얻을 수 있습니다.

훈련 손실만 낮고 테스트 손실이 높다면 과적합을 의심해야 합니다. 테스트 데이터의 결과를 보고 모델을 계속 조정하면 테스트 데이터에 간접적으로 맞추게 되므로, 실제 실험에서는 검증 데이터도 별도로 두는 것이 좋습니다.

## 9. 시각화

`keras10_scatter*.py`에서는 다음을 사용합니다.

```python
plt.scatter(x, y)
plt.plot(x, results, color='red')
plt.show()
```

- `scatter`: 실제 관측값을 점으로 표시
- `plot`: 모델 예측 흐름을 선으로 표시

숫자 지표만 보는 것보다 데이터의 분포, 이상치, 예측선의 방향을 직관적으로 확인할 수 있습니다.

## 10. 실제 데이터셋 회귀

`keras11_*`에서는 California Housing, Diabetes, Boston Housing 데이터를 사용합니다.

실제 데이터는 입력 특성 수와 값의 범위가 다양하고, 샘플 수가 많으며, 단순 예제보다 노이즈가 있습니다. 따라서 단순히 학습 데이터에서 loss가 작다는 것만으로 좋은 모델이라고 판단하면 안 됩니다.

- California Housing: 입력 8개
- Diabetes: 입력 10개
- Boston Housing: 입력 13개

각 데이터셋의 `x.shape`, `y.shape`, 입력 차원, train/test 분할이 모델의 첫 레이어와 맞아야 합니다.

## 11. R2, MSE, RMSE

`keras12_*`에서 여러 평가 지표를 비교합니다.

### MSE

오차 제곱의 평균입니다. 작을수록 좋습니다. 단위가 실제값의 제곱이 됩니다.

### RMSE

MSE에 제곱근을 적용합니다.

```python
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
```

실제값과 같은 단위를 가지므로 오차 크기를 직관적으로 이해하기 쉽습니다. 작을수록 좋습니다.

### R2

모델이 정답의 변동을 얼마나 설명하는지 나타내는 지표입니다. 일반적으로 1에 가까울수록 좋고, 0이면 평균값으로 예측하는 수준과 비슷하며, 음수가 될 수도 있습니다.

MSE·RMSE·R2는 서로 다른 관점을 제공하므로 하나만 보고 판단하지 않습니다.

## 12. Pandas와 따릉이 데이터

`keras13_ddarung.py`는 CSV 데이터를 읽고 결측치를 처리한 뒤 회귀 모델에 넣습니다.

```python
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
```

- `index_col=0`: 첫 번째 열을 index로 사용
- `dropna()`: 결측치가 있는 행 삭제
- `drop(['count'], axis=1)`: `count` 열을 제거해 입력 X 생성
- `train_csv['count']`: 정답 y 생성

훈련 데이터에서 `dropna()`를 사용하면 결측 행이 제거되지만, 테스트 데이터는 제출 대상이므로 무조건 행을 삭제하면 안 됩니다. 테스트 데이터의 행 수가 달라지면 제출 파일의 행과 대응하지 않기 때문입니다. 실제 프로젝트에서는 평균·중앙값 대체, 모델 기반 대체 등 다른 결측치 처리 방법을 검토해야 합니다.

## 13. 가장 중요한 연결 관계

```text
x.shape[1] == 첫 Dense의 input_dim
y.shape[1] == 마지막 Dense의 출력 노드 수  # y가 2차원인 경우
x와 y의 첫 번째 축 크기 == 샘플 수
train_test_split(x, y)에서는 X와 Y의 행 대응 유지
평가 지표는 학습 데이터가 아닌 별도 테스트 데이터로 확인
```

## 14. validation과 과적합 대응

`keras16`, `keras17`은 train/test 외에 validation 데이터를 사용하는 방법을 다룹니다.

- train: 가중치 학습
- validation: 학습 중 모델·하이퍼파라미터 선택
- test: 최종 일반화 성능 평가

직접 `x_val`, `y_val`을 나눌 수도 있고, `model.fit(..., validation_split=0.2)`로 학습 데이터 일부를 validation으로 자동 분리할 수도 있습니다. test 데이터를 모델 선택에 계속 사용하면 test 성능에 맞춰지는 문제가 생기므로 validation을 따로 두는 것이 중요합니다.

`keras18`은 학습 시간을 측정하고, `keras19`는 `history`와 matplotlib로 train loss와 validation loss를 시각화해 과적합을 확인합니다. train loss는 계속 내려가는데 validation loss가 다시 올라가면 과적합 신호입니다.

`keras20`은 `EarlyStopping`을 사용합니다. validation 성능이 더 좋아지지 않을 때 학습을 중단해 불필요한 epoch와 과적합을 줄입니다. `monitor`, `patience`, `restore_best_weights` 설정을 함께 이해해야 합니다.

## 15. 회귀와 분류

`keras21`부터는 분류 문제를 다룹니다. 문제 유형에 따라 마지막 레이어와 손실 함수를 바꿉니다.

### 이진 분류

- 출력층: `Dense(1, activation='sigmoid')`
- 손실: `binary_crossentropy`
- 예측 확률을 임계값으로 클래스에 변환
- 평가: accuracy 등

유방암과 Santander 예제가 이 구조를 사용합니다. `stratify=y`는 train/test 분할 때 클래스 비율을 유지하기 위해 사용합니다.

### 다중 분류

- 정답을 one-hot 형태로 변환
- 출력층: 클래스 수만큼의 `Dense(..., activation='softmax')`
- 손실: `categorical_crossentropy`
- 예측 결과는 각 클래스 확률 벡터

Iris, Wine, Covtype, Digits 예제가 이 구조를 사용합니다. 마지막 출력 노드 수는 클래스 수와 같아야 하며, one-hot 정답의 열 수와도 일치해야 합니다.

## 16. 스케일링

`keras27`, `keras28`, `keras52`에서는 특성의 크기와 분포를 조정합니다.

- `MinMaxScaler`: 일정 범위로 변환
- `StandardScaler`: 평균 0, 표준편차 1에 가깝게 변환
- `MaxAbsScaler`: 절댓값 최대치를 기준으로 변환
- `RobustScaler`: 중앙값과 사분위 범위를 사용해 이상치에 비교적 강함

중요한 원칙은 scaler를 train 데이터에만 `fit`하고, test에는 같은 scaler로 `transform`만 하는 것입니다.

```python
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
```

test까지 포함해 `fit`하면 test 정보가 학습 과정에 섞이는 데이터 누수가 발생합니다.

## 17. 모델 저장과 체크포인트

`keras29`는 모델 전체 또는 가중치만 저장·복원합니다.

- `model.save(...)`: 구조·가중치·compile 정보 등을 포함한 전체 모델
- `load_model(...)`: 저장된 전체 모델 복원
- `save_weights(...)`: 가중치만 저장
- `load_weights(...)`: 동일하거나 호환되는 구조에 가중치 적용

`keras30`, `keras31`은 `ModelCheckpoint`를 사용해 학습 중 성능이 좋은 모델을 파일로 보존합니다. 전체 모델 저장과 weights-only 저장은 복원할 때 필요한 코드와 파일 확장자가 다르므로 구분해야 합니다.

## 18. Dropout과 optimizer

`keras33`, `keras34`는 `Dropout`을 이용한 정규화를 다룹니다. Dropout은 학습 중 일부 뉴런을 임시로 비활성화해 특정 경로에 과도하게 의존하는 것을 줄입니다. 평가·예측 시에는 일반적으로 자동으로 비활성화됩니다.

`keras52`는 `Adam(learning_rate=...)`처럼 learning rate를 직접 조정합니다. learning rate가 너무 크면 손실이 불안정하고, 너무 작으면 학습이 느릴 수 있습니다. 성능 비교 시 epoch만 늘리지 말고 데이터 스케일·optimizer·learning rate·검증 손실을 함께 봐야 합니다.

## 19. CNN과 이미지 데이터

`keras36`부터 CNN을 사용합니다.

- `Conv2D`: 이미지의 지역 패턴 추출
- `MaxPooling2D` 또는 `MaxPool2D`: 공간 크기 축소와 중요한 특징 선택
- `Dropout`: 과적합 완화
- `Flatten`: CNN 출력을 1차원으로 변환
- `GlobalAveragePooling2D`: 공간 위치별 평균으로 요약해 파라미터를 줄임
- 마지막 `sigmoid` 또는 `softmax`: 분류 클래스 수에 맞춘 출력

입력 shape는 보통 `(높이, 너비, 채널)`입니다.

- MNIST/Fashion-MNIST: `(28, 28, 1)`
- CIFAR10/100: `(32, 32, 3)`
- 컬러 사진 예제: `(100, 100, 3)` 또는 `(200, 200, 3)`

Conv2D의 kernel 크기와 pooling을 거치면 높이·너비가 줄어듭니다. 각 층 뒤의 shape가 다음 층의 입력과 맞는지 확인해야 합니다.

## 20. 이미지 파일 처리와 증강

`keras45`부터 이미지를 NumPy 배열로 변환해 저장하고 불러옵니다. `keras48`은 `load_img`, `img_to_array`로 한 장의 이미지를 모델 입력용 배열로 바꾸는 과정을 보여줍니다. 학습한 모델을 불러온 뒤 새 이미지를 같은 전처리와 shape로 맞춰 예측해야 합니다.

`keras50`, `keras51`은 `ImageDataGenerator`로 회전·이동·확대·반전 등의 변형을 만들어 학습 데이터 다양성을 높입니다. 증강은 일반적으로 train 데이터에 적용하고, validation/test는 실제 평가를 위해 원래 분포를 유지해야 합니다.

## 21. 전체 범위 핵심 선택표

| 문제 | 출력층 | 손실 함수 | 대표 평가 |
| --- | --- | --- | --- |
| 연속값 1개 회귀 | `Dense(1)` | `mse` | MSE, RMSE, R2 |
| 이진 분류 | `Dense(1, sigmoid)` | `binary_crossentropy` | accuracy |
| 다중 분류 | `Dense(클래스 수, softmax)` | `categorical_crossentropy` | accuracy |
| 이미지 이진 분류 | CNN + `Dense(1, sigmoid)` | `binary_crossentropy` | accuracy |
| 이미지 다중 분류 | CNN + `Dense(클래스 수, softmax)` | `categorical_crossentropy` | accuracy |

## 22. 최초 복습 문제

### 기본 개념

1. `Sequential` 모델은 어떤 방식으로 레이어를 구성합니까?
2. `fit`, `evaluate`, `predict`의 차이를 각각 설명하세요.
3. `loss='mse'`가 의미하는 것을 수식과 말로 설명하세요.
4. `epochs=100`과 `batch_size=4`는 각각 무엇을 뜻합니까?
5. `optimizer='adam'`의 역할은 무엇입니까?

### shape와 모델 구조

6. `x.shape == (100, 8)`일 때 샘플 수, 특성 수, `input_dim`은 각각 얼마입니까?
7. `y.shape == (100, 3)`인 다중 출력 회귀에서 마지막 Dense 레이어의 노드 수는 얼마여야 합니까?
8. `x = np.array([[1, 2, 3], [4, 5, 6]])`의 shape는 무엇입니까? 이 배열을 transpose하면 shape가 어떻게 바뀝니까?
9. `x.T`를 사용했을 때 샘플과 특성의 방향이 의도대로 바뀌었는지 무엇을 확인해야 합니까?
10. `input_dim`을 실제 특성 수와 다르게 지정하면 어떤 문제가 생깁니까?

### 데이터 분리와 일반화

11. 학습 데이터와 테스트 데이터를 분리하는 이유는 무엇입니까?
12. `train_test_split(x, y)`에 X와 Y를 함께 넘겨야 하는 이유는 무엇입니까?
13. `random_state`는 무엇을 보장합니까?
14. train loss는 낮은데 test loss가 높다면 어떤 현상을 의심해야 합니까?
15. 테스트 데이터를 반복해서 확인하며 모델 구조를 계속 바꾸면 어떤 문제가 생길 수 있습니까?

### 평가와 전처리

16. MSE와 RMSE의 차이는 무엇입니까? RMSE가 해석하기 쉬운 이유는 무엇입니까?
17. R2가 1에 가까운 경우와 0인 경우를 각각 설명하세요. R2가 음수가 될 수도 있는 이유는 무엇입니까?
18. `dropna()`로 훈련 데이터의 결측 행을 제거할 때 생길 수 있는 문제는 무엇입니까?
19. 테스트 데이터의 결측 행을 무조건 삭제하면 제출 데이터에서 어떤 문제가 생깁니까?
20. 따릉이 코드에서 `count`를 X에서 제거하고 y로 분리하는 이유는 무엇입니까?

### 응용 문제

21. 입력 특성이 9개이고 출력이 1개인 회귀 모델의 최소 입출력 구조를 코드로 작성하세요.
22. 실제값과 예측값이 있을 때 RMSE를 계산하는 코드를 작성하세요.
23. 모델 성능을 평가할 때 MSE, RMSE, R2를 함께 보는 이유를 설명하세요.
24. 단순한 모델과 깊은 모델 중 어느 쪽이 항상 더 좋은지, 그렇지 않다면 왜 그런지 설명하세요.
25. 현재 저장소의 학습 흐름을 `데이터 → 모델 → 학습 → 평가` 순서로 5문장 이내에 요약하세요.

### 확장 범위 문제

26. train, validation, test 데이터를 각각 언제 사용하며 서로 바꾸면 안 되는 이유는 무엇입니까?
27. train loss는 내려가고 validation loss는 올라갈 때 어떤 현상이며, `EarlyStopping`은 어떻게 도움을 줍니까?
28. 이진 분류와 다중 분류에서 출력층과 손실 함수를 각각 작성하세요.
29. `sigmoid`와 `softmax`의 출력값은 각각 어떤 의미를 가집니까?
30. `stratify=y`를 사용하는 이유는 무엇입니까?
31. `StandardScaler`를 test 데이터에 다시 `fit`하면 안 되는 이유를 데이터 누수 관점에서 설명하세요.
32. `model.save()`와 `model.save_weights()`의 차이를 설명하세요.
33. `ModelCheckpoint`는 어떤 상황에서 유용합니까?
34. Dropout이 학습 중과 평가 중에 다르게 동작하는 이유는 무엇입니까?
35. Conv2D, MaxPooling2D, Flatten의 역할을 순서대로 설명하세요.
36. 입력 이미지 shape가 `(32, 32, 3)`이라는 것은 무엇을 의미합니까?
37. CNN에서 pooling을 사용하면 공간 크기와 특징 표현에 어떤 변화가 생깁니까?
38. `GlobalAveragePooling2D`와 `Flatten`의 차이를 설명하세요.
39. `ImageDataGenerator`를 train 데이터에는 적용하고 test 데이터에는 조심해야 하는 이유는 무엇입니까?
40. learning rate가 너무 크거나 너무 작을 때 학습에 어떤 문제가 생길 수 있습니까?
41. 회귀 문제에 sigmoid 출력층과 categorical crossentropy를 그대로 사용하면 안 되는 이유는 무엇입니까?
42. 이미지 파일을 `load_img`와 `img_to_array`로 변환한 뒤 모델에 넣기 전에 확인해야 할 전처리는 무엇입니까?
43. CPU와 GPU 학습 시간 비교에서 시간만 보고 판단하면 안 되는 이유는 무엇입니까?

## 답변 방법

번호를 붙여 아는 만큼 답변하면 됩니다. 모르는 문제는 `모름`이라고 적어도 됩니다. 답변을 분석해 다음 복습에서 부족한 부분을 우선 출제합니다.
