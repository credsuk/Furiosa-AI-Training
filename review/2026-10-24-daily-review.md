# 2026-10-24 AI 주말 복습 문제

대상 범위: 2026-10-24 학습 코드 없음 → 최근 학습일 기준 `LG/lg07_ToolNode_graph01.py`·`02.py`, `LG/lg08_Memory01.py`·`02.py`, `LG/lg06_02_ddg.py`(2026-10-20~21)와 Python·NumPy·pandas 기초
비고: 2026-10-24(토)에는 새 학습 코드가 없어, 가장 최근에 다루지 않은 주제인 **도구 호출(ToolNode)과 메모리(checkpointer)** 를 중심으로 냈습니다. 2026-10-23 문제(트리 모델 평가·병렬 병합)와 겹치지 않게 구성했습니다.
문제 수: 10문제 (객관식 5문제, 주관식 5문제)

## 객관식

### 1. 아래 그래프에서 `chatbot` 노드가 돌려준 답변에 `tool_calls`가 **없을 때** `tools_condition`은 어디로 보낼까요?

```python
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
```

- A. `"tools"` 노드로 보낸다
- B. `END`로 보내서 그래프를 끝낸다
- C. `START`로 되돌려 처음부터 다시 시작한다
- D. 오류를 낸다. `tool_calls`가 없으면 조건 함수가 결과를 못 정하기 때문이다

### 2. `lg08_Memory01.py`는 첫 질문에서 "내 이름은 김기관이야"라고 말했지만, 두 번째 질문 "내 이름이 뭐라고 했지?"에는 제대로 답하지 못합니다. 가장 알맞은 이유는 무엇일까요?

- A. `temperature=0`이라서 모델이 대답을 못 한다
- B. `compile()`에 checkpointer를 넘기지 않아 대화가 저장되지 않고, 매번 처음 상태에서 시작하기 때문이다
- C. `TavilySearch(max_results=3)`의 검색 결과가 3개뿐이라서
- D. `State`에 `messages` 키가 없어서

### 3. `lg07_ToolNode_graph01.py`에서 `png_data`는 `draw_mermaid_png()`가 돌려준 **bytes**입니다. 아래 코드의 `'wb'`를 `'w'`로 바꾸면 어떻게 될까요?

```python
with open(path + "lg07_ToolNode.png", 'wb') as file:
    file.write(png_data)
```

- A. 똑같이 잘 저장된다
- B. `TypeError`가 난다. `'w'`는 글자(str)를 쓰는 모드인데 bytes를 쓰려 했기 때문이다
- C. 파일은 만들어지지만 그림이 깨져서 열리지 않는다
- D. `'w'`는 읽기 모드라서 파일이 만들어지지 않는다

### 4. 다음 코드의 출력은 무엇일까요?

```python
import numpy as np
a = np.arange(6).reshape(2, 3)
print(a.sum(axis=0))
```

- A. `[3 12]`
- B. `[3 5 7]`
- C. `15`
- D. `(2, 3)`

### 5. 다음 pandas 코드의 두 출력으로 맞는 것은?

```python
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
print(df["b"].mean())
print(df.shape)
```

- A. `20.0` 그리고 `(3, 2)`
- B. `20.0` 그리고 `(2, 3)`
- C. `60` 그리고 `(3, 2)`
- D. `10.0` 그리고 `(3,)`

## 주관식

### 6. 도구를 쓰는 챗봇의 흐름을 순서대로 설명해 보세요.

아래 코드를 기준으로, "사용자가 질문 → … → 최종 답변"까지 노드가 어떤 순서로 실행되는지, 그리고 **도구가 필요할 때와 필요 없을 때** 길이 어떻게 갈리는지 적어 보세요.

```python
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
graph = builder.compile()
```

1. `chatbot` 노드는 무엇을 하나요?
2. `tools_condition`은 무엇을 보고 길을 정하나요?
3. `tools` 노드(`ToolNode`)는 무슨 일을 하나요?
4. 도구를 두 번 연달아 써야 하는 질문이라면 이 고리는 어떻게 돌까요?

### 7. `lg08_Memory01.py`와 `lg08_Memory02.py`는 거의 같습니다. 다른 곳을 찾아 보고, 왜 02만 대화를 기억하는지 설명해 보세요.

1. 두 파일에서 서로 다른 줄을 **세 곳** 찾아 적어 보세요.
2. `thread_id`는 무슨 역할을 하나요? `"user01"` 대신 `"user02"`로 바꾸면 어떻게 될까요?
3. `InMemorySaver`라는 이름에서 알 수 있는 한계는 무엇인가요?

### 8. 코드의 `os.environ["MONOROUTER_API_KEY"].strip()`에는 왜 `.strip()`이 붙어 있을까요?

1. `.strip()`이 하는 일을 정확히 적어 보세요.
2. `.strip()`을 빼면 어떤 문제가 생길 수 있나요? (키를 복사·붙여넣기 했을 때를 생각해 보세요)
3. `.strip()`으로도 해결되지 않는 경우는 무엇인가요?

### 9. 그래프 그림을 저장하고 보여 주는 코드를 설명해 보세요.

```python
png_data = graph.get_graph().draw_mermaid_png()

with open("./_save/image/lg07_ToolNode.png", 'wb') as file:
    file.write(png_data)
```

```python
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(BytesIO(png_data))
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

1. `'wb'`에서 `b`는 무슨 뜻인가요?
2. `with`를 쓰면 무엇이 편해지나요? `with` 없이 쓰면 어떤 실수를 하기 쉬울까요?
3. `BytesIO(png_data)`는 왜 필요한가요? `Image.open("파일경로")`와 무엇이 다른가요?
4. `figsize=(12, 8)`, `imshow`, `axis('off')`는 각각 무엇을 하나요?

### 10. 아래 코드의 세 출력을 적고, 왜 그렇게 나오는지 설명해 보세요.

```python
scores = [70, 80, 90]

total = 0
for s in scores:
    total += s
print(total, total / len(scores))

import numpy as np
a = np.array(scores)
print(a.sum(), a.mean())

import pandas as pd
df = pd.DataFrame({"score": scores})
print(df["score"].sum(), df["score"].mean())
```

1. 세 `print`의 출력을 각각 적어 보세요.
2. `a.mean()`의 결과에 소수점이 붙는 이유는 무엇인가요? (70+80+90은 240이라 딱 나누어떨어지는데도요)
3. `df["score"]`의 dtype은 무엇일까요? 정수 데이터를 넣었기 때문에 어떻게 될지 생각해 보세요.

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요.
- 이번 문제는 코드의 흐름과 계산 결과를 묻습니다. 실제 LLM 호출, 웹 검색, 그래프 창 실행 결과를 묻는 문제는 없습니다.
- NumPy·pandas 문제(4, 5, 10번)의 숫자는 실제 실행으로 확인한 값입니다.
