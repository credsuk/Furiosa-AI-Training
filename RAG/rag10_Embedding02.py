import _llm

prompt = "삼성전자의 창업주는 누구인가요?"

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", # 임베딩을 해주는 모델 끝날때까지 이거쓰면 충분
    api_key=_llm.api_key,
    base_url=_llm.base_url,
    dimensions=5,  # 디멘션 조절이 가능하다 / 최대치보다 크게하면 400에러 발생
)

vector = embeddings.embed_query(prompt)
print(vector)
print("======================================")
print("임베딩 벡터의 차원 : ", len(vector)) # 임베딩 벡터의 차원 :  5


# _llm 패키지 업데이트
# vector = _llm.getEmbedding(prompt)

