# 2026-09-29 AI 학습 일일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 확인 시각: 2026-09-29 20:01 KST
- 오늘 커밋: `90a261e` — 공개 저장소에서 복습 자료를 제거하고 Hermes 내부 기억 경로로 이동
- 오늘 Python 변경: 없음
- 오늘 복습 주제: 최근 `day18`의 시계열 모델 요약, `SimpleRNN`·`GRU`·`LSTM`, 입력 shape, 시계열 window 분할, 다중 feature 예측

## 오늘 읽은 코드에서 확인한 핵심

1. `SimpleRNN`, `GRU`, `LSTM`은 시계열 입력을 `(샘플 수, timestep, feature 수)` 형태로 받는다. 단일 시계열을 `x.reshape(x.shape[0], x.shape[1], 1)`로 바꾸면 마지막 축이 feature 1개가 된다.
2. `keras54_RNN2_summary.py`는 `(7, 3, 1)` 입력에서 `SimpleRNN(10)` 뒤에 Dense 층을 연결한다. `keras55_LSTM1_summary.py`는 같은 문제에서 GRU를 사용하며, `keras55_LSTM2_scale.py`는 LSTM으로 바꿔 큰 값이 포함된 시계열을 예측한다.
3. `keras56_split1.py`의 window 분할은 길이 `size`인 구간을 만들고, 각 구간의 마지막 값을 y로 분리해 앞부분으로 다음 값을 예측하는 구조다. 코드에는 학습 전에 `exit()`가 있어 뒤의 모델 학습 부분은 실행되지 않는다.
4. `keras56_split2.py`는 `(10, 2)`인 다중 feature 시계열을 만들고, 입력 shape를 `(샘플, 4, 2)`로 구성한다. y에서 두 번째 feature만 선택해 예측한다.
5. 검증에는 `validation_split`, `EarlyStopping(restore_best_weights=True)`, `ReduceLROnPlateau`를 사용한다. 다만 매우 작은 데이터에서 `validation_split=0.01`은 검증 샘플 수가 지나치게 작을 수 있으므로 결과 해석에 주의해야 한다.

## 헷갈리기 쉬운 부분

- timestep과 feature의 순서를 혼동하지 않기: `(batch, timestep, feature)`이며 `input_shape`에는 batch를 넣지 않는다.
- window의 입력 길이와 예측 target의 위치를 정확히 추적하기.
- `SimpleRNN`·`GRU`·`LSTM`의 파라미터 수와 내부 gate 차이를 단순히 배수로 외우지 않기.
- 시계열에서는 무작위 분할보다 시간 순서를 보존하는 평가가 필요한 경우가 많다.
- 모델을 `x, y` 전체로 평가하면 학습 성능을 보는 것이므로 일반화 성능과 혼동하지 않기.

## 문제 (총 7문제)

### 객관식 1 — 입력 shape

다음 코드에서 `model.add(LSTM(15, input_shape=(4, 2)))`가 의미하는 입력 구조로 가장 알맞은 것은 무엇입니까?

A. 샘플마다 timestep 2개, feature 4개
B. 샘플마다 timestep 4개, feature 2개
C. 전체 데이터가 4개 샘플이고 feature가 2개
D. 출력이 4개이고 입력 feature가 2개

### 객관식 2 — 코드 실행 결과

`a = np.array(range(1, 11))`, `size = 5`일 때 `split_x(a, size)`가 반환하는 배열의 shape는 무엇입니까?

A. `(5, 5)`
B. `(6, 4)`
C. `(6, 5)`
D. `(10, 5)`

### 객관식 3 — 올바른 코드 선택

길이 5인 window에서 앞의 4개를 입력 X, 마지막 1개를 정답 y로 만들려 할 때 가장 적절한 코드는 무엇입니까?

A. `x = bbb[:, 1:]`; `y = bbb[:, :1]`
B. `x = bbb[:, :-1]`; `y = bbb[:, -1:]`
C. `x = bbb[:-1, :]`; `y = bbb[-1:, :]`
D. `x = bbb[:, -1:]`; `y = bbb[:, :-1]`

### 객관식 4 — O/X 및 오류 원인

다음 설명은 맞습니까? “시계열 입력이 `(샘플, timestep, feature)`일 때 `input_shape`에는 샘플 수까지 포함해 `input_shape=(샘플, timestep, feature)`로 작성해야 한다.”

A. O — 샘플 수를 포함해야 한다.
B. X — `input_shape`에는 `(timestep, feature)`만 넣는다.
C. O — 단, LSTM일 때만 그렇다.
D. X — `input_shape`에는 feature 수만 넣는다.

### 주관식 5 — 개념 설명

`SimpleRNN`, `GRU`, `LSTM`이 모두 순서가 있는 데이터를 처리하는 데 사용되는 이유를 설명하고, GRU 또는 LSTM이 기본 RNN의 어떤 문제를 완화하려는지 짧게 쓰세요.

### 주관식 6 — 코드 작성

1차원 배열 `data`와 정수 `timestep`을 받아, 각 샘플의 입력은 길이 `timestep`, 정답은 그 다음 한 값이 되도록 `x`, `y`를 반환하는 `split_xy(data, timestep)` 함수를 작성하세요. 반환 shape도 함께 적으세요.

### 주관식 7 — 응용·오류 수정

`keras56_split2.py`처럼 입력이 `(샘플, 4, 2)`인 다중 feature 시계열에서 두 번째 feature의 다음 값을 예측하려고 합니다. `y = y[:, 0, 1]`이 하는 일을 설명하고, 시간 순서를 섞지 않고 평가하려면 데이터 분할에서 무엇을 주의해야 하는지 쓰세요.

## 답변 방법

번호를 붙여 1~7번을 답하세요. 객관식은 선택지 문자와 이유를, 주관식은 코드 또는 설명을 함께 적으면 됩니다. 모르는 문제는 `모름`이라고 적어도 됩니다.
