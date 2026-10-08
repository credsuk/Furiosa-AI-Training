# 2026-10-08 AI 학습 평일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-08 (목요일)
- 기준 최신 학습 코드 SHA: `22ac6f6de1a27d09646aaa625ab3e840106a221c` (`day26`)
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)
- 범위: PDF/RAG/임베딩/FAISS, Keras 다중 입력·다중 출력·Concatenate, Python·NumPy·pandas 기초 연결

## 객관식

### 1. RAG의 기본 순서

RAG 챗봇의 일반적인 흐름으로 가장 알맞은 것은 무엇인가요?

- **A.** 질문을 먼저 삭제하고 LLM만 실행한다.
- **B.** 문서를 읽고 쪼갠 뒤 임베딩·저장하고, 질문과 가까운 내용을 검색해 답변에 참고한다.
- **C.** PDF를 이미지로만 바꾼 뒤 아무 검색 없이 답한다.
- **D.** FAISS가 문장을 직접 한글 답변으로 바꾼다.

### 2. NumPy 전치와 shape

```python
x1 = np.array([range(100), range(301, 401)]).T
```

`x1.shape`는 무엇인가요?

- **A.** `(2, 100)`
- **B.** `(100, 2)`
- **C.** `(100, 1, 2)`
- **D.** `(200,)`

### 3. Concatenate 결과

첫 번째 가지의 출력 shape가 `(batch, 80)`, 두 번째 가지의 출력 shape가 `(batch, 40)`일 때 다음 결과는 무엇인가요?

```python
merge = Concatenate()([output1, output2])
```

- **A.** `(batch, 40)`
- **B.** `(batch, 80)`
- **C.** `(batch, 120)`
- **D.** `(batch, 80, 40)`

### 4. 안전한 역직렬화 판단

`FAISS.load_local(..., allow_dangerous_deserialization=True)`에 대한 설명으로 가장 알맞은 것은 무엇인가요?

- **A.** 모든 파일을 안전하게 만들어 주는 옵션이다.
- **B.** pickle 역직렬화를 허용하므로 신뢰할 수 있는 파일에만 사용해야 한다.
- **C.** 임베딩 차원을 자동으로 0으로 만든다.
- **D.** Gradio 화면을 자동으로 실행한다.

## 주관식

### 5. 다중 입력 모델의 샘플 대응

`x1.shape == (100, 2)`, `x2.shape == (100, 3)`, `y.shape == (100,)`인 데이터를 다중 입력 모델에 넣는다고 합시다.

1. 첫 번째 숫자 100이 뜻하는 것을 설명하세요.
2. `train_test_split(x1, x2, y, random_state=213)`처럼 함께 나누어야 하는 이유를 쓰세요.
3. `model.fit([x1_train, x2_train], y_train)`에서 두 입력의 샘플 순서가 어긋나면 어떤 문제가 생기는지 설명하세요.

### 6. 다중 출력과 shape

`keras69_ensemble3.py`처럼 모델이 `outputs=[last_output1, last_output2]`를 가지고, `y1`, `y2`를 각각 예측한다고 합시다.

1. `fit()`에 정답을 `[y1_train, y2_train]` 순서로 넣어야 하는 이유를 설명하세요.
2. 각 출력이 `Dense(1)`이면 일반적으로 출력 shape는 어떻게 되나요?
3. 코드에서 `last_dense1`, `last_dense2`를 만들었지만 최종 출력에 연결하지 않았다면 그 층은 현재 계산 경로에 포함되나요? 이유를 쓰세요.

### 7. Python·NumPy·pandas 코드 검토

다음 상황을 초보자에게 설명하세요.

```python
x = np.array([range(6), range(10, 16)]).T
train_df = df.drop([target], axis=1)
```

1. `x.shape`를 쓰고 `.T`가 한 일을 설명하세요.
2. pandas에서 `axis=1`은 무엇을 뜻하는지 설명하세요.
3. 오늘 RAG/앙상블 코드만으로 실제 학습 loss, 임베딩 벡터 차원, Gradio 서버 실행 성공을 단정할 수 없는 이유를 검증 범위와 연결해 쓰세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·데이터 대응·검증 범위를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
