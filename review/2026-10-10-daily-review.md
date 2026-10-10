# 2026-10-10 AI 학습 주말 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-10 (토요일)
- 기준 최신 학습 코드 SHA: `22ac6f6de1a27d09646aaa625ab3e840106a221c` (`day26`)
- 당일 새 학습 코드 commit: 없음
- 문제 구성: 총 10문제 (객관식 5, 주관식 5)
- 범위: RAG/PDF/임베딩/FAISS/Gradio, Keras 다중 입력·다중 출력·Concatenate, Python·NumPy·pandas 기초

## 객관식

### 1. RAG의 전체 흐름

RAG 챗봇의 흐름으로 가장 알맞은 것은 무엇인가요?

- **A.** 문서를 읽고 작은 조각으로 나눈 뒤 임베딩·저장하고, 질문과 가까운 조각을 찾아 답변에 참고한다.
- **B.** PDF를 삭제한 다음 LLM만 실행한다.
- **C.** Gradio가 벡터 검색과 임베딩을 모두 대신한다.
- **D.** FAISS가 검색하지 않고 항상 정답을 직접 만들어 준다.

### 2. NumPy 전치와 shape

```python
x = np.array([range(6), range(10, 16)]).T
```

`x.shape`는 무엇인가요?

- **A.** `(2, 6)`
- **B.** `(6, 2)`
- **C.** `(12,)`
- **D.** `(6, 1, 2)`

### 3. 다중 입력의 feature 수

`x1.shape == (100, 2)`, `x2.shape == (100, 3)`일 때 올바른 설명은 무엇인가요?

- **A.** 샘플 2개와 3개를 뜻한다.
- **B.** 두 배열 모두 샘플 100개이며, feature 수가 각각 2개와 3개이다.
- **C.** 첫 번째 배열은 timestep 100개, 두 번째 배열은 timestep 3개라는 뜻이다.
- **D.** 두 배열을 합치려면 샘플 수가 달라야 한다.

### 4. Concatenate 결과 shape

`output1.shape == (batch, 80)`, `output2.shape == (batch, 40)`일 때 `Concatenate()([output1, output2])`의 shape는 무엇인가요?

- **A.** `(batch, 40)`
- **B.** `(batch, 80)`
- **C.** `(batch, 120)`
- **D.** `(batch, 80, 40)`

### 5. 안전한 FAISS 로딩

`allow_dangerous_deserialization=True`에 대한 설명으로 알맞은 것은 무엇인가요?

- **A.** 어떤 pickle 파일도 안전하게 만들어 준다.
- **B.** 역직렬화를 허용하므로 출처를 믿을 수 있는 파일에만 사용해야 한다.
- **C.** 임베딩 모델을 자동으로 다운로드한다.
- **D.** Gradio 서버를 자동으로 외부 공개한다.

## 주관식

### 6. 여러 입력과 샘플 대응

`x1.shape == (100, 2)`, `x2.shape == (100, 3)`, `y.shape == (100,)`이라고 합시다.

1. 각 숫자가 뜻하는 것을 설명하세요.
2. 세 배열을 `train_test_split`에 함께 넣어야 하는 이유를 설명하세요.
3. 샘플 순서가 어긋나면 어떤 문제가 생기는지 쓰세요.

### 7. Keras 다중 출력의 순서

모델이 `outputs=[last_output1, last_output2]`이고 정답 배열이 `y1`, `y2`라면 `fit()`과 `evaluate()`에 어떤 순서로 정답을 넣어야 하나요? 순서가 바뀌면 왜 문제가 되나요?

### 8. 연결되지 않은 층 찾기

다음 코드에서 `last_dense1`, `last_dense2`가 실제 최종 출력 계산에 사용되는지 설명하세요.

```python
last_dense1 = Dense(10, name="ld1")(merge3)
last_dense2 = Dense(10, name="ld2")(last_dense1)
last_output1 = Dense(1, name="last1")(merge3)
```

사용되게 만들려면 `last_output1`을 어떻게 연결해야 할까요?

### 9. RAG 구성요소의 역할

`PyPDFLoader`, 임베딩 모델, FAISS, retriever, prompt의 `{context}`, Gradio `ChatInterface`가 각각 어떤 일을 하는지 순서대로 설명하세요. `k=2` 또는 `k=3`은 무엇을 정하는 값인가요?

### 10. 실행 결과와 코드 주석 구분

다음 두 종류의 주장을 구분해서 설명하세요.

1. 코드 주석에 적힌 `loss`, `predict`, 임베딩 벡터 차원 숫자를 보고 현재 환경에서도 그대로 재현된다고 말하는 것
2. TensorFlow 학습, 로컬 임베딩 모델 로딩, FAISS 검색, OpenAI 호출, Gradio 실행을 실제로 하지 않고도 모두 성공했다고 말하는 것

또한 아래 코드의 결과를 설명하세요.

```python
train_df = df.drop([target], axis=1)
```

`axis=1`은 무엇을 뜻하나요?

## 답변 방법

1~10번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·샘플 대응·연결 경로·실제 검증 여부를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
