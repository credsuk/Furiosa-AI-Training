# 2026-09-30 AI 학습 일일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main` (실행 시 확인된 로컬 `origin/main`: `c5ed6c0`)
- 기준 날짜: 2026-09-30 (수요일, 평일)
- 근거 학습 commit: `29f740b` (`day19`)
- 주제: 시계열 window, 다중 feature shape, `return_sequences`, Jena 기후 데이터
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)

## 복습 핵심

- `split_x`의 window 개수는 `len(data) - size + 1`이다.
- 입력 shape는 `(batch, timestep, feature)`이고 Keras `input_shape`에는 batch를 쓰지 않는다.
- `[:, -1:]`처럼 슬라이스로 축을 보존하는 것과 `[:, -1]`로 축을 제거하는 것을 구별한다.
- `return_sequences=True`는 다음 recurrent layer에 모든 timestep 출력을 전달할지 정하는 shape 설정이다.
- 시계열 평가에서는 시간 순서를 보존하며, scaler는 train에만 `fit`하고 test에는 `transform`한다.
- Python 함수의 변수명·import·반환값과 pandas DataFrame/Series/NumPy 변환도 함께 확인한다.

## 객관식

### 1. Window 개수와 실제 shape

`keras56_split3.py`의 `a = np.array(range(1, 101))`, `size = 6` 및 `range(len(dataset) - size + 1)`을 기준으로 `bbb = split_x(a, size)`의 shape는 무엇인가요? `bbb[:, :-1]`, `bbb[:, -1:]`의 shape도 함께 고르세요.

- **A.** `bbb (94, 6)`, X `(94, 5)`, y `(94, 1)`
- **B.** `bbb (95, 6)`, X `(95, 5)`, y `(95, 1)`
- **C.** `bbb (95, 5)`, X `(95, 4)`, y `(95, 1)`
- **D.** `bbb (100, 6)`, X `(100, 5)`, y `(100, 1)`

선택한 이유와 파일의 주석 출력이 계산 결과와 맞는지도 적어 보세요.

### 2. 다중 feature target indexing

`bbb.shape == (45, 6, 2)`일 때 아래 코드의 결과 shape는 무엇인가요?

```python
y = bbb[:, -1:, 1]
```

- **A.** `(45, 6, 2)`
- **B.** `(45, 1, 2)`
- **C.** `(45, 1)`
- **D.** `(1, 45)`

선택한 이유를 축별로 설명해 보세요.

### 3. `return_sequences`의 목적

여러 LSTM/GRU 층을 연결할 때 중간 recurrent 층에서 `return_sequences=True`를 설정하는 주된 이유는 무엇인가요?

- **A.** 학습 데이터가 자동으로 섞이는 것을 막기 위해
- **B.** 모든 timestep의 출력을 다음 recurrent 층에 전달하기 위해
- **C.** 모델의 loss를 자동으로 낮추기 위해
- **D.** 입력 feature 축을 제거하기 위해

선택한 이유를 출력 tensor의 축과 연결해 설명해 보세요.

### 4. 시계열 검증과 스케일링

시간 순서가 중요한 데이터에 대한 설명으로 가장 적절한 것은 무엇인가요?

- **A.** train/test를 무작위로 섞고 scaler는 전체 데이터에 `fit`한다.
- **B.** 시간 순서를 보존해 분할하고 scaler는 train에 `fit`, test에 `transform`한다.
- **C.** test 데이터를 먼저 scaler에 `fit`하고 train에 `transform`한다.
- **D.** `shuffle=False`이면 test 구간도 train 데이터로 사용해도 된다.

선택한 이유와 데이터 누수 위험을 설명해 보세요.

## 주관식

### 5. Python 함수 작성과 반환 shape

NumPy를 사용해 1차원 `data`와 `timestep`을 입력받고, 각 X가 연속된 `timestep`개 값, 각 y가 바로 다음 한 값이 되도록 `split_xy(data, timestep)`을 작성하세요. 입력 길이가 `n`일 때 X와 y의 shape도 적고, 필요한 import와 반복 범위를 확인하세요.

### 6. pandas 처리 흐름 설명

Jena 예제에서 `pd.read_csv(path, index_col=0)`의 반환 객체는 무엇인가요? 이어서 `.drop(['wd (deg)'], axis=1).to_numpy(dtype=np.float32)`는 어떤 열을 제거하고 최종적으로 어떤 자료형의 데이터를 반환하는지 설명하세요. 이번 예제에서 확인한 실제 동작과 일반적인 pandas 참고 지식을 구분해 주세요.

### 7. 코드 검토: 입력 shape와 예측 누수

`keras56_split3.py`에서 `bbb`는 `(95, 6)`이고 `x = bbb[:, :-1]`의 shape는 `(95, 5)`입니다. 그런데 모델은 `input_shape=(5, 1)`로 선언되어 있습니다. 모델 입력에 필요한 shape와 현재 코드에서 빠진 변환을 설명하세요. 이어서 Jena 예제의 train/test 순서 보존 및 train-only scaler fit이 왜 필요한지, `evaluate(x, y)`를 학습 데이터 전체에 실행한 점수가 무엇을 검증하는지 설명하세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 코드·shape·설명을 포함해 주세요. 모르는 부분은 `모름`이라고 적어도 됩니다.
