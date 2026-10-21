# 2026-10-21 AI 일일 복습 문제

대상 범위: `LG/lg07_ToolNode_graph01.py` ~ `LG/lg09_병렬실행_병합2.py`  
문제 수: 7문제 (객관식 4문제, 주관식 3문제)

## 객관식

### 1. `draw_mermaid_png()`의 결과를 파일에 저장할 때 사용하는 모드는 무엇인가요?

- A. `r`
- B. `w`
- C. `wb`
- D. `ab`

### 2. `InMemorySaver`와 함께 같은 대화 흐름을 이어 보려면 무엇을 같게 유지해야 하나요?

- A. 함수 이름
- B. `thread_id`
- C. PNG 파일 이름
- D. 모델의 `temperature`

### 3. 다음 그래프가 두 작업을 동시에 시작하게 만드는 연결은 무엇인가요?

```text
START → search
START → analysis
```

- A. 시작점에서 두 노드로 각각 엣지를 연결한다.
- B. `merge` 노드를 두 번 실행한다.
- C. `END`를 두 개 만든다.
- D. 리스트를 문자열로 바꾼다.

### 4. 다음 NumPy 코드에서 `arr.shape`는 무엇인가요?

```python
import numpy as np
arr = np.array([[1, 2], [3, 4], [5, 6]])
```

- A. `(2, 3)`
- B. `(3, 2)`
- C. `(6,)`
- D. `(1, 6)`

## 주관식

### 5. `InMemorySaver`와 `thread_id`가 대화 기억을 이어 주는 과정을 쉬운 말로 설명해 보세요. 프로그램을 종료했을 때도 영구 보존되는지 함께 적어 보세요.

### 6. 병렬 실행 그래프에서 `search`와 `analysis`의 결과를 `merge`가 합치는 과정을 설명해 보세요. 두 결과를 상태에 저장할 때 사용할 수 있는 키 이름도 적어 보세요.

### 7. 다음 pandas 코드에서 `axis=0`과 `axis=1`의 차이를 설명하고, 각각의 합 결과를 적어 보세요.

```python
import pandas as pd

df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
print(df.sum(axis=0))
print(df.sum(axis=1))
```

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요.
- 오늘 문제는 실제 API 응답이나 화면 출력 결과가 아니라 코드 구조와 데이터 흐름을 묻습니다.
