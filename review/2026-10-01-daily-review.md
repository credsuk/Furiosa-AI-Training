# 2026-10-01 AI 학습 일일 복습

- 기준 저장소: `Furiosa-AI-Training`
- 기준 브랜치: `main`
- 대상 학습일: 2026-10-01 (목요일)
- 기준 최신 SHA: `68b28c01569c95f6804c7e60b17efa88a6a4b146` (`day21`, 17:45 KST)
- 참고 commit: `77fd6b0` 및 `68b28c0` (`day21`)
- 주제: 토큰화·padding·one-hot encoding·Embedding·텍스트 분류·RAG 임베딩
- 문제 구성: 총 7문제 (객관식 4, 주관식 3)

## 복습 핵심

- `Tokenizer`는 단어를 정수 ID로 바꾸며, `texts_to_sequences` 결과는 문장마다 길이가 다를 수 있습니다.
- `pad_sequences`는 시퀀스를 같은 길이로 만들며 `padding`은 빈자리, `truncating`은 초과 토큰을 자를 위치를 정합니다.
- 정수 토큰을 `Embedding`에 넣으면 `(batch, timestep)`이 `(batch, timestep, output_dim)`으로 바뀝니다.
- `OneHotEncoder`와 `to_categorical`은 입력 형태·사용 목적이 다르므로 데이터와 라벨의 역할을 구분해야 합니다.
- 임베딩 API 호출은 외부 credential과 네트워크가 필요하므로 코드 정적 확인과 실제 API 실행 결과를 구분합니다.

## 객관식

### 1. Tokenizer와 시퀀스

`token.fit_on_texts(docs)` 후 `token.texts_to_sequences(docs)`의 일반적인 결과로 맞는 것은 무엇인가요?

- **A.** 모든 문장이 동일한 길이의 실수 벡터가 된다.
- **B.** 단어가 정수 ID로 바뀌며 문장별 리스트 길이는 다를 수 있다.
- **C.** 단어가 자동으로 one-hot 행렬로 바뀐다.
- **D.** 원문 문자열이 그대로 반환된다.

### 2. padding과 truncating

다음 설정의 의미로 맞는 것은 무엇인가요?

```python
pad_sequences(x, maxlen=5, padding='pre', truncating='pre')
```

- **A.** 짧은 시퀀스의 앞을 0으로 채우고, 긴 시퀀스의 앞을 잘라 최대 5개로 만든다.
- **B.** 짧은 시퀀스의 뒤를 0으로 채우고, 긴 시퀀스의 뒤를 잘라낸다.
- **C.** 모든 토큰을 삭제하고 길이 5인 0 배열만 만든다.
- **D.** padding과 truncating은 모두 모델의 출력 shape만 바꾼다.

### 3. Embedding shape

정수 토큰 입력 `padded_x.shape == (15, 5)`에 `Embedding(input_dim=31, output_dim=10)`을 적용했을 때, Embedding 직후의 shape로 맞는 것은 무엇인가요?

- **A.** `(15, 5)`
- **B.** `(15, 10)`
- **C.** `(15, 5, 10)`
- **D.** `(31, 10, 15, 5)`

### 4. RAG 임베딩 API

`OpenAIEmbeddings(...).embed_query(prompt)`의 반환값에 대한 설명으로 가장 적절한 것은 무엇인가요?

- **A.** 입력 문장의 정수 token ID 리스트만 반환한다.
- **B.** 입력 문장을 수치 벡터로 변환한 리스트를 반환하며 차원은 모델 설정·API 결과에 따른다.
- **C.** 반드시 길이 5인 one-hot 벡터를 반환한다.
- **D.** 검색된 문서 원문을 자동으로 반환한다.

## 주관식

### 5. Python·NumPy shape 추적

다음 코드에서 `padded_x`의 shape가 `(15, 5)`이고 `output_dim=10`일 때, Embedding과 `SimpleRNN(10)` 각각의 출력 shape를 적으세요. `batch`, `timestep`, `feature`가 각 축에서 무엇을 뜻하는지도 설명하세요.

```python
model.add(Embedding(input_dim=31, output_dim=10))
model.add(SimpleRNN(10))
```

### 6. 코드 수정: padding과 예측 입력

학습 입력은 `pad_sequences(x, padding='pre', maxlen=5, truncating='pre')`로 만들었는데, 예측 입력에는 `maxlen=3`을 사용했습니다. 이 코드가 왜 문제가 될 수 있는지 설명하고, 학습과 예측에 같은 전처리 규칙을 적용하도록 수정 예시를 작성하세요. `Tokenizer`에 학습 어휘에 없는 단어가 들어올 때 생길 수 있는 문제도 적으세요.

### 7. one-hot과 Embedding 선택·검증 범위

정수 토큰 입력을 one-hot으로 바꾸는 방식과 Embedding 레이어를 사용하는 방식의 차이를 설명하세요. 이어서 `RAG/rag10_Embedding01.py`의 임베딩 API 호출을 실제 실행하지 못할 때, 무엇을 정적으로 검증할 수 있고 무엇은 `확인 불가`로 남겨야 하는지 적으세요. credential 값은 답변에 쓰지 마세요.

## 답변 방법

1~7번 번호를 붙여 답하세요. 객관식은 선택지와 이유를, 주관식은 shape·코드·검증 범위를 적어 주세요. 모르는 항목은 `모름`이라고 적어도 됩니다.
