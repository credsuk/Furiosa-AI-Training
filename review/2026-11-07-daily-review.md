# 2026-11-07 AI 주말 복습 문제

- 기준 저장소/브랜치: `Furiosa-AI-Training` / `main`
- 기준 SHA: `b043cfb896679bf8a6e0c57f8bc1a741d7e3a79f` (`day46`)
- 다룬 범위: `day45`·`day46`의 Python 모듈, 모델 선택, Grid/Random Search, Keras 회귀 탐색
- 구성: 10문제 (객관식 5, 주관식 5). 11월 마지막 주말이 아니므로 주간 복습 기준입니다.
- 모델 학습·코드 실행은 수행하지 않았으며, 후보 수 계산과 Python AST 검사를 수행했습니다.

## 객관식

### 1. `GridSearchCV`와 `RandomizedSearchCV`의 차이를 가장 정확히 설명한 것은 무엇인가요?

- A. `GridSearchCV`는 지정한 모든 조합을 시험하고, `RandomizedSearchCV`는 정한 횟수만큼 조합을 뽑아 시험한다.
- B. `GridSearchCV`는 분류 전용이고, `RandomizedSearchCV`는 회귀 전용이다.
- C. `RandomizedSearchCV`는 모델을 학습하지 않고 숫자만 무작위로 출력한다.
- D. 두 도구 모두 언제나 파라미터 한 조합만 시험한다.

### 2. 다음 세 파라미터 사전의 후보 수와 `cv=5`에서 필요한 교차검증 fit 수는 얼마인가요?

```python
parameters = [
    {'n_estimators': [100, 500], 'max_depth': [6, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]},
    {'max_depth': [6, 8, 10, 12], 'learning_rate': [0.1, 0.01, 0.001]},
    {'min_child_weight': [2, 3, 5, 10], 'learning_rate': [0.1, 0.01, 0.001]},
]
```

- A. 후보 12개, fit 60회
- B. 후보 42개, fit 210회
- C. 후보 234개, fit 1,170회
- D. 후보 210개, fit 42회

### 3. `load_diabetes()`의 연속형 `y`를 `XGBRFRegressor`로 학습할 때, 5-fold 분할 방법으로 가장 적절한 것은 무엇인가요?

- A. `StratifiedKFold` — 연속형 정답을 자동으로 같은 개수의 등급으로 바꿔 준다.
- B. `KFold` — 회귀 자료를 fold로 나누고, 평가는 R²·MAE 같은 회귀 지표로 한다.
- C. `StratifiedKFold`와 `accuracy_score` — 모든 회귀 예측은 정확도다.
- D. `train_test_split`을 매번 test에 적용하고 그 점수로 최적 모델을 고른다.

### 4. `load_digits()`의 정답은 숫자 클래스인데, 코드가 `XGBRFRegressor`와 기본 `.score()`를 사용합니다. 의도한 숫자 분류 성능을 평가하려면 무엇이 맞나요?

- A. `XGBClassifier`와 분류용 교차검증·정확도를 사용한다.
- B. `XGBRFRegressor`를 유지하고 R²를 정확도라고 부른다.
- C. `StratifiedKFold`를 없애고 test 점수로 파라미터를 반복 선택한다.
- D. `load_digits()`의 정답을 pandas 열 이름으로 바꾼다.

### 5. 다음 코드에서 `p11_car.py`의 `drive()`가 최상위에서 호출되고, `p13_do.py`가 import 뒤 `p11_car.drive()`도 호출합니다. `p13_do.py`를 직접 실행하면 어떤 일이 일어나나요?

```python
# p11_car.py
def drive():
    print('운전하다')
drive()  # 최상위 문장

# p13_do.py
import p11_car
p11_car.drive()
```

- A. `drive()`가 한 번만 실행된다. import할 때는 파일의 최상위 코드가 실행되지 않는다.
- B. `drive()`가 두 번 실행된다. 첫 호출은 import 중 최상위 문장, 두 번째는 명시적 함수 호출이다.
- C. `drive()` 함수는 import하면 자동으로 삭제된다.
- D. `__name__`이 두 파일에서 모두 `__main__`이 된다.

## 주관식

### 6. NumPy와 pandas 기초

```python
scores = np.array([0.55, 0.81, 0.74])
index = np.argmax(scores)

# df.shape == (80, 3), columns = ['temp', 'target', 'wind']
features = df.drop(['target'], axis=1)
```

1. `index` 값은 무엇인가요? 이 값은 최고 점수 자체인가요, 점수가 있던 위치인가요?
2. `features.shape`는 무엇이며, `axis=1`은 무엇을 제거하나요?
3. `np.array([range(6), range(10, 16)]).T.shape`는 무엇인가요? 전치는 축을 어떻게 바꾸며, 차원을 하나 더 만들까요?

### 7. test 데이터 누수 방지

아래 코드가 여러 설정 중 최선의 것을 고르는 데 `x_test`, `y_test`를 사용하면 왜 문제가 되나요? `best_score_`와 마지막 test 점수의 역할을 구분해 설명해 보세요.

추가로 `x1.shape == (100, 2)`, `x2.shape == (100, 3)`, `y.shape == (100,)`이라면 세 배열을 `train_test_split` 한 번에 함께 넣는 이유는 무엇인가요? 행 순서를 따로 섞으면 어떤 위험이 있나요?

```python
for params in parameter_candidates:
    model.set_params(**params)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
```

### 8. 전처리와 교차검증 고치기

아래 흐름에서 scaler 누수를 막도록 핵심 코드를 고쳐 쓰세요. 회귀 `y`를 5-fold로 평가한다고 가정합니다.

```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
scores = cross_val_score(estimator, x_train, y_train, cv=5)
```

`Pipeline`이 fold마다 무엇을 다시 학습하는지도 설명해 보세요.

### 9. `ml07_for로파라미터튜닝.py`에서 고칠 점 찾기

이 파일은 `load_diabetes()`의 연속형 정답을 `HistGradientBoostingClassifier`에 넣고, 여러 설정의 점수를 같은 `x_test`, `y_test`로 비교합니다. 또 마지막에 `"{:.2f}".format(best_parameters)`를 사용합니다.

문제점 두 가지 이상과 각각의 수정 방향을 적어 주세요. 가능하면 모델 종류·평가 데이터 사용·마지막 출력 타입을 모두 검토해 보세요.

### 10. Keras 회귀 하이퍼파라미터 탐색

`keras2/keras77_hyperParameter_02_diabetes.py`를 참고해 답해 보세요.

1. 입력 shape `(442, 10)`과 출력 `Dense(1, activation='linear')`는 왜 회귀 문제에 맞나요?
2. `RandomizedSearchCV(n_iter=10, cv=3)`은 교차검증 fit을 몇 번 수행하나요? 기본 `refit=True`의 의미는 무엇인가요?
3. `build_model()`은 `lr=0.001`을 인자로 받습니다. 이 값이 optimizer에 적용되거나 `create_hyperparameteer()`의 검색 후보에 들어가나요? 들어가지 않는다면 실제 학습률 탐색에 어떤 영향이 있나요? 정적 검토만으로 실행 결과를 단정할 수 없는 이유도 한 가지 적어 주세요.

## 답변 방법

- 객관식은 `1. A`처럼 번호와 선택지를 적고, 가능하면 이유도 써 주세요.
- 주관식은 번호를 붙이고 짧은 설명이나 코드를 적어 주세요. 계산 문제는 계산 과정을 함께 쓰면 어디서 혼동했는지 확인하기 쉽습니다.
- 모르는 문제는 비워 두어도 됩니다.
