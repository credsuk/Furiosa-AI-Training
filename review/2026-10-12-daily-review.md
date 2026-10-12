# 2026-10-12 AI 학습 평일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-12 (월요일)
- 기준 최신 학습 코드 SHA: `98d640356dff2fac62e4a3faba001ab0d8b0c84e` (`day27`)
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)
- 범위: RAG 대화 기록·MessagesPlaceholder·Gradio, Keras 다중 출력, 시계열 window·스케일링·다중 입력, Python·NumPy·pandas 기초

## 객관식

### 1. 대화 기록 자리

`MessagesPlaceholder(variable_name='chat_history')`의 주된 역할은 무엇인가요?

- **A.** 대화 메시지 목록이 들어갈 자리를 prompt에 마련한다.
- **B.** FAISS 인덱스를 자동으로 저장한다.
- **C.** CSV의 날짜 열을 삭제한다.
- **D.** Keras 모델의 loss를 계산한다.

### 2. 다중 출력 정답

`model = Model(outputs=[last_output1, last_output2])`인 모델을 학습할 때 가장 알맞은 정답 전달 방법은 무엇인가요?

- **A.** `fit(x, y1)`만 전달한다.
- **B.** `fit(x, [y1, y2])`처럼 출력 순서와 같은 목록을 전달한다.
- **C.** `fit([y1, y2], x)`로 입력과 정답을 뒤집는다.
- **D.** 두 정답을 문자열로 합친다.

### 3. 시계열 window

길이 1120인 데이터에서 `split_size=20`으로 `split_xy()`를 만들면 window 개수는 얼마인가요?

- **A.** 20개
- **B.** 1100개
- **C.** 1101개
- **D.** 1120개

### 4. scaler 사용 순서

테스트 데이터의 정보가 학습에 미리 섞이는 것을 줄이는 올바른 방법은 무엇인가요?

- **A.** train과 test를 합쳐 scaler를 `fit()`한다.
- **B.** test만으로 scaler를 `fit()`한다.
- **C.** train으로 `fit()`하고 train/test에는 각각 `transform()`한다.
- **D.** scaler를 사용하지 않고 임의의 숫자를 뺀다.

## 주관식

### 5. RAG 대화 흐름

`reg21_Gradio_history.py`에서 Gradio의 `history` 항목을 `HumanMessage`와 `AIMessage`로 바꾸는 이유를 설명하세요. `retriever`, `chat_history`, 현재 `input`이 답변에 어떻게 함께 사용되는지도 쓰세요.

### 6. 계산 그래프

`keras69_ensemble4.py`에서 `last_dense1`, `last_dense2`를 만들었는데도 최종 출력에 영향을 주지 않는 이유를 설명하세요. 두 층을 실제로 사용하려면 `last_output1`을 어떻게 연결해야 하나요?

### 7. 시계열 입력과 데이터 누수

주가 코드의 입력 shape `(batch, 20, 6)`에서 각 숫자가 뜻하는 것을 설명하고, scaler를 train 데이터에만 `fit()`해야 하는 이유를 쉬운 예로 설명하세요. 시간 순서를 무작위로 섞지 않는 이유도 함께 쓰세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 코드 흐름·shape·실제 검증 범위를 포함해 설명하세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
