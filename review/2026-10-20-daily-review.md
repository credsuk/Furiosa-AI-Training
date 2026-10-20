# 2026-10-20 AI 일일 복습 문제

대상 범위: `LG/lg05_ToolNode.py` ~ `LG/lg07_ToolNode_graph.py`  
문제 수: 7문제 (객관식 4문제, 주관식 3문제)

## 객관식

### 1. `llm.bind_tools(tools=tools)`의 주된 역할은 무엇인가요?

- A. 검색 결과를 자동으로 파일에 저장한다.
- B. 모델이 사용할 수 있는 도구 정보를 연결한 모델 객체를 만든다.
- C. LangGraph 그래프를 자동으로 컴파일한다.
- D. 환경변수의 API 키를 새로 발급한다.

### 2. `ToolNode(tools)`가 하는 일로 가장 알맞은 것은 무엇인가요?

- A. 모델의 tool call에 따라 등록된 실제 도구를 실행한다.
- B. `TypedDict`의 모든 값을 정수로 변환한다.
- C. 그래프의 모든 엣지를 삭제한다.
- D. PNG 파일의 shape를 계산한다.

### 3. 다음 그래프에서 검색 결과가 chatbot으로 돌아가는 이유는 무엇인가요?

```text
START → chatbot → tools → chatbot
```

- A. `tools → chatbot` 엣지가 도구 결과를 모델에 다시 전달하기 때문이다.
- B. `END`가 모든 노드를 무한히 반복시키기 때문이다.
- C. `add_messages`가 인터넷 검색을 대신하기 때문이다.
- D. `TypedDict`가 자동으로 모델을 두 번 호출하기 때문이다.

### 4. 다음 NumPy 코드의 `arr.shape`는 무엇인가요?

```python
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
```

- A. `(6,)`
- B. `(2, 3)`
- C. `(3, 2)`
- D. `(1, 6)`

## 주관식

### 5. `bind_tools`와 `ToolNode`의 차이를 “도구를 알려 주는 단계”와 “도구를 실행하는 단계”라는 표현을 사용해 설명해 보세요.

### 6. 다음 Python 코드의 출력 결과와 이유를 설명해 보세요.

```python
messages = ["질문", "검색 결과", "최종 답변"]
print(messages[-1])
```

### 7. 다음 pandas 코드에서 `axis=1`은 어느 방향으로 계산하라는 뜻인지 설명하고, 각 행의 합을 구한 결과를 적어 보세요.

```python
import pandas as pd

df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
print(df.sum(axis=1))
```

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요.
- 이번 문제는 외부 API의 실제 검색 결과를 묻지 않습니다. 코드 구조와 실행 흐름을 설명하는 것이 목표입니다.
