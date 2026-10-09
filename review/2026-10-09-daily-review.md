# 2026-10-09 AI 학습 평일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-09 (금요일)
- 기준 최신 학습 코드 SHA: `22ac6f6de1a27d09646aaa625ab3e840106a221c` (`day26`)
- 당일 새 학습 코드 commit: 없음
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)
- 범위: RAG/PDF/임베딩/FAISS, Keras 다중 입력·다중 출력·Concatenate, Python·NumPy·pandas 기초

## 객관식

### 1. RAG 순서

RAG 챗봇의 흐름으로 가장 알맞은 것은 무엇인가요?

- **A.** 문서를 읽고 쪼갠 뒤 임베딩·저장하고, 질문과 가까운 내용을 검색해 답변에 참고한다.
- **B.** PDF를 지우고 LLM만 실행한다.
- **C.** FAISS가 검색 없이 정답 문장을 직접 만든다.
- **D.** Gradio가 문서 임베딩을 대신한다.

### 2. NumPy shape

```python
x = np.array([range(6), range(10, 16)]).T
```

`x.shape`는 무엇인가요?

- **A.** `(2, 6)`
- **B.** `(6, 2)`
- **C.** `(12,)`
- **D.** `(6, 1, 2)`

### 3. Concatenate

`output1.shape == (batch, 80)`, `output2.shape == (batch, 40)`일 때 `Concatenate()([output1, output2])`의 shape는 무엇인가요?

- **A.** `(batch, 40)`
- **B.** `(batch, 80)`
- **C.** `(batch, 120)`
- **D.** `(batch, 80, 40)`

### 4. 역직렬화 옵션

`allow_dangerous_deserialization=True`에 대한 설명으로 알맞은 것은 무엇인가요?

- **A.** 모든 외부 파일을 안전하게 만든다.
- **B.** pickle 역직렬화를 허용하므로 신뢰할 수 있는 파일에만 사용해야 한다.
- **C.** 임베딩 벡터를 자동으로 삭제한다.
- **D.** Gradio 서버를 실행한다.

## 주관식

### 5. 다중 입력의 샘플 대응

`x1.shape == (100, 2)`, `x2.shape == (100, 3)`, `y.shape == (100,)`이라고 합시다.

1. 각 숫자가 뜻하는 것을 설명하세요.
2. 세 배열을 `train_test_split`에 함께 넣어야 하는 이유를 설명하세요.
3. 샘플 순서가 어긋나면 모델 학습에 어떤 문제가 생기는지 쓰세요.

### 6. 다중 출력의 순서와 연결

모델이 `outputs=[last_output1, last_output2]`이고 정답이 `y1`, `y2`라면 `fit()`에 어떤 순서로 정답을 넣어야 하나요? 또한 중간 Dense 층을 만들기만 하고 최종 출력에 연결하지 않으면 어떻게 되는지 설명하세요.

### 7. 검증 범위와 기초 배열

다음 코드의 결과와 검증 범위를 설명하세요.

```python
x = np.array([range(6), range(10, 16)]).T
train_df = df.drop([target], axis=1)
```

1. `.T` 뒤의 shape를 쓰세요.
2. `axis=1`의 뜻을 쓰세요.
3. 코드만 읽고 실제 loss, 임베딩 차원, Gradio 실행 성공을 단정하면 안 되는 이유를 쓰세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·샘플 대응·실제 검증 여부를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
