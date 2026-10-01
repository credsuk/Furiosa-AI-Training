import _llm

prompt = "삼성전자의 창업주는 누구인가요?"

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", # 임베딩을 해주는 모델 끝날때까지 이거쓰면 충분
    api_key=_llm.api_key,
    base_url=_llm.base_url,
)

vector = embeddings.embed_query(prompt)
print(vector)
print("======================================")
print("임베딩 벡터의 차원 : ", len(vector)) # 임베딩 벡터의 차원 :  1536
