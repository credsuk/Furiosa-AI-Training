# 2026-10-19 AI 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-19 (월요일)
- 기준 SHA: `fd2e56b19d4d6c15595a75af4c8dd989ca8b40f7`
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)
- 범위: `TypedDict`, `Annotated`, `add_messages`, LangGraph 상태·노드·엣지·실행, Python·NumPy·pandas 기초

## 객관식

### 1. TypedDict의 역할

`Person(TypedDict)`를 작성했을 때 가장 알맞은 설명은 무엇인가요?

- **A.** 딕셔너리가 어떤 키와 타입을 예상하는지 알려 주는 타입 설계도다.
- **B.** 모든 값을 자동으로 문자열로 바꾼다.
- **C.** 딕셔너리를 자동으로 데이터베이스에 저장한다.
- **D.** API 키를 자동으로 암호화한다.

### 2. add_messages의 ID 처리

`add_messages`에 기존 메시지와 같은 `id`를 가진 새 메시지를 전달하면 일반적으로 어떤 일이 일어나나요?

- **A.** 같은 ID의 기존 메시지가 새 메시지로 교체된다.
- **B.** 프로그램이 항상 종료된다.
- **C.** 모든 메시지가 숫자로 바뀐다.
- **D.** 새 메시지가 무조건 버려진다.

### 3. LangGraph 연결

`START → prepare → chatbot → finish → END` 흐름을 만드는 데 필요한 것은 무엇인가요?

- **A.** 노드와 노드 사이를 연결하는 엣지
- **B.** pandas의 `DataFrame`만
- **C.** NumPy 배열의 `reshape`만
- **D.** `print` 문만

### 4. Python·NumPy·pandas 기초

다음 중 올바른 설명은 무엇인가요?

- **A.** `enumerate`는 순번과 항목을 함께 꺼내고, NumPy의 `shape`는 축별 크기를 나타내며, pandas `DataFrame`은 행과 열을 가진 표 형태 자료다.
- **B.** `enumerate`는 API 키를 숨기고, `shape`는 원소의 합을 뜻한다.
- **C.** DataFrame에는 열이 없고 하나의 문자열만 저장된다.
- **D.** NumPy 배열은 여러 축의 크기를 표현할 수 없다.

## 주관식

### 5. 그래프 실행 순서

LangGraph에서 `State`, 노드, 엣지, `compile()`, `invoke()`가 각각 어떤 역할을 하는지 쉬운 말로 설명하고, `START → chatbot → END`의 실행 순서를 적으세요.

### 6. 메시지 상태 병합

서로 다른 ID를 가진 사용자·AI 메시지와 같은 ID를 가진 메시지를 `add_messages`에 넣었을 때 결과가 어떻게 달라지는지 설명하세요. 단순한 `append`와 다른 점도 적으세요.

### 7. 모델 호출과 안전한 설정

`chatbot(state)`가 `state["messages"]`를 모델에 전달하는 흐름을 설명하세요. API 키와 base URL을 코드에 직접 출력하지 않고 환경변수에서 읽어야 하는 이유와, 이 코드에서 아직 실제로 확인하지 못한 항목 한 가지를 적으세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 단계별 흐름과 쉬운 예시를 포함해 설명하세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
