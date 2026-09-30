# 2026-09-30 AI 학습 일일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-09-30 (수요일)
- 기준 최신 SHA: `4b1aa938cb3b819da46be095c320992b87e1bf9d` (`day20`, 17:44 KST; 20시 cron 전에 반영됨)
- 참고 학습 commit: `29f740b` (`day19`), `4b1aa93` (`day20`)
- 주제: 시계열 window/shape, Bidirectional RNN, Jena 온도 예측, LangChain·LCEL
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)

## 복습 핵심

- `split_x` window 개수는 `len(data) - size + 1`이고, `-1`과 `-1:`은 결과 축을 다르게 만든다.
- 시계열 입력은 `(batch, timestep, feature)`이며 Keras `input_shape`에는 batch 축을 적지 않는다.
- `Bidirectional`은 recurrent layer를 양방향으로 감싸며, 기본 concat에서는 두 방향의 units가 합쳐진다.
- LCEL은 `prompt | model | output_parser`처럼 구성 요소를 연결한다.
- 실제 코드의 target 열과 API 자격 증명 처리는 선언·주석만 믿지 말고, 학습과 예측 경로 및 안전한 비밀정보 처리를 각각 확인한다.

## 객관식

### 1. Bidirectional 출력 shape

`x.shape == (7, 3, 1)`일 때 아래 층을 기본 설정으로 사용하고 `return_sequences`를 지정하지 않으면 출력 shape는 무엇인가요?

```python
Bidirectional(SimpleRNN(16), input_shape=(3, 1))
```

- **A.** `(7, 3, 16)`
- **B.** `(7, 16)`
- **C.** `(7, 32)`
- **D.** `(7, 3, 32)`

기본 merge 방식과 두 방향의 출력을 연결해 이유를 설명하세요.

### 2. Window와 NumPy shape

길이 100인 1차원 배열에서 `size=6`으로 다음 window 함수를 사용합니다.

```python
for i in range(len(data) - size + 1):
    windows.append(data[i:i + size])
```

`windows`, `x = windows[:, :-1]`, `y = windows[:, -1:]`의 shape 조합은 무엇인가요?

- **A.** `(94, 6)`, `(94, 5)`, `(94, 1)`
- **B.** `(95, 6)`, `(95, 5)`, `(95, 1)`
- **C.** `(95, 5)`, `(95, 4)`, `(95, 1)`
- **D.** `(100, 6)`, `(100, 5)`, `(100, 1)`

### 3. pandas → NumPy 변환

Jena 예제에서 `datasets`가 `pd.read_csv(..., index_col=0)`의 결과이고 `T (degC)` 열을 제거한다고 할 때, 다음 표현의 역할로 맞는 것은 무엇인가요?

```python
datasets.drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
```

- **A.** 행을 제거하고 pandas Series를 반환한다.
- **B.** 열을 제거하고 지정 dtype의 NumPy 배열을 반환한다.
- **C.** 열을 제거하고 DataFrame의 index를 반환한다.
- **D.** 결측치를 제거하고 Python list를 반환한다.

### 4. LCEL 구성

프롬프트에서 모델을 호출한 뒤 출력 문자열을 얻는 체인으로 가장 적절한 것은 무엇인가요?

- **A.** `model | prompt | parser`
- **B.** `prompt | model | StrOutputParser()`
- **C.** `parser | prompt | model`
- **D.** `prompt + model + parser`

## 주관식

### 5. Python 함수와 window 반환 shape

NumPy 배열 `data`와 정수 `timestep`을 받아 각 입력 window의 길이는 `timestep`, 정답은 바로 다음 값 한 개가 되도록 `split_xy(data, timestep)`을 작성하세요. `len(data) == n`일 때 X와 y의 shape를 적고, import·변수명·반복 범위도 확인하세요.

### 6. Helper 연결 오류 찾기

`RAG/_llm.py`에는 `setPrompt(...)` 함수가 정의되어 있습니다. `RAG/rag08_LCEL02.py`의 호출부가 다른 함수 이름을 사용한다면 실행 시 어떤 종류의 오류가 발생할 수 있나요? 호출을 어떻게 맞추면 되는지 답하고, `invoke`에서 prompt·model·parser가 어떤 순서로 연결되는지 설명하세요.

### 7. 데이터 누수·입력 일관성·보안

`keras59_Bidirectional3_jena.py`에서 학습용 `x_data`는 `T (degC)`를 제거하지만 예측용 `x_predict`는 `wd (deg)`를 제거합니다. 어떤 입력 일관성 문제가 생길 수 있는지 설명하세요. 이어서 왜 시계열 train/test 분할에서 순서를 보존하는지, 코드에 API key를 직접 쓰거나 일부를 출력하는 방식 대신 어떤 안전한 처리 원칙을 써야 하는지 적어 보세요. 자격 증명 값은 답변에 쓰지 마세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 코드·shape·설명을 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
