# 2026-10-22 AI 일일 복습 문제

대상 범위: `LG/lg09_병렬실행_병합2_concat.py`, `LG/lg09_병합실행_병합3.py`, `ml/ml01_feature_importtances01.py`~`04_그래프.py`
문제 수: 7문제 (객관식 4문제, 주관식 3문제)

## 객관식

### 1. 병렬 LangGraph 예제에서 `search`와 `analysis`가 각자 만든 결과를 `merge` 노드가 읽으려면 무엇이 필요할까요?

- A. 두 결과를 같은 상태 키에 덮어쓴다.
- B. 서로 다른 상태 키에 저장하고 `merge`가 그 키를 읽는다.
- C. `END`에서 두 값을 만든다.
- D. `thread_id`를 두 개 만든다.

### 2. 두 결과를 LLM에게 전달해 최종 답변을 새로 작성하게 하는 코드는 어느 방식인가요?

- A. 두 문자열을 `+`로 이어 붙인다.
- B. `llm.invoke(prompt)`로 두 결과가 담긴 프롬프트를 보낸다.
- C. `plt.imshow()`를 호출한다.
- D. `train_test_split()`을 호출한다.

### 3. 다음 NumPy 배열의 `shape`는 무엇인가요?

```python
import numpy as np
x = np.array([[5, 8, 2], [1, 4, 7]])
```

- A. `(2, 3)`
- B. `(3, 2)`
- C. `(6,)`
- D. `(1, 6)`

### 4. 분류 데이터에서 `train_test_split(x, y, stratify=y)`를 사용하는 주된 이유는 무엇인가요?

- A. 평가 데이터를 없애기 위해
- B. 나누기 전후의 정답 클래스 비율을 비슷하게 유지하기 위해
- C. 특징 열의 이름을 자동으로 붙이기 위해
- D. 회귀 점수를 F1-score로 바꾸기 위해

## 주관식

### 5. pandas에서 아래 두 합계의 차이를 말하고, 각 결과를 적어 보세요.

```python
import pandas as pd
df = pd.DataFrame({"a": [2, 3], "b": [10, 20]})
print(df.sum(axis=0))
print(df.sum(axis=1))
```

### 6. `feature_importances_`, XGBoost의 `gain`, F1-score가 서로 어떤 점에서 다른지 쉬운 말로 설명해 보세요. `feature_importances_`를 F1-score라고 불러도 되는지도 답해 보세요.

### 7. 병렬 검색·분석 그래프에서 `search_result`, `analysis_result`가 최종 답변으로 바뀌는 과정을 두 방식(문자열 연결과 LLM 종합)으로 나누어 설명해 보세요. 각 방식의 차이도 한 가지 적어 보세요.

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요.
- 오늘 문제는 코드 구조와 데이터 흐름을 묻습니다. 외부 API나 모델을 실제 실행한 결과를 묻는 문제가 아닙니다.
