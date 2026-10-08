# rag10_Embedding01 복사

import _llm

prompt = "삼성전자의 창업주는 누구인가요?"

# pip install sentence-transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    # 베이징 어느 대학교에서 나온건데 말귀를 잘 알아 듣는 모델이다 (한국, 중국, 일본은 아주 좋음)
    model_name="BAAI/bge-m3", 
    model_kwargs={
        "device" : "cpu", # GPU : cuda / CPU : cpu
        "local_files_only" : True, # 다운로드 받지말고 먼저 받아둔 이미지를 사용해라
    }
)

vector = embeddings.embed_query(prompt)
# print(vector)
print("======================================")
print("임베딩 벡터의 차원 : ", len(vector)) # 임베딩 벡터의 차원 :  1024





