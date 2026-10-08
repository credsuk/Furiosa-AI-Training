# reg20_embedding2_bge-m3 복사

import _llm

prompt = "삼성전자의 창업주는 누구인가요?"

# pip install sentence-transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    # 베이징 어느 대학교에서 나온건데 말귀를 잘 알아 듣는 모델이다 (한국, 중국, 일본은 아주 좋음)
    model_name="Qwen/Qwen3-Embedding-0.6B", 
    model_kwargs={
        "device" : "cpu", # GPU : cuda / CPU : cpu
        "local_files_only" : True,
    },
    encode_kwargs = {"normalize_embeddings": True}
)

vector = embeddings.embed_query(prompt)
# print(vector)
print("======================================")
print("임베딩 벡터의 차원 : ", len(vector)) # 임베딩 벡터의 차원 :  1024





