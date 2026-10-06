import os, _llm
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma

# ChromaDB 불러오기
DB_PAHT = "./_db/Chroma11/"
db = Chroma(
    embedding_function=_llm.getEmbedding(),
    persist_directory=DB_PAHT, # 경로
    collection_name='croma11', # 이름 / 쉽게 테이블 이름
)

# 저장된 데이터 확인
print("==========================================================")
print(db.get())

print("==========================================================")
# 한국말로 유사도 검색, 코싸인 유사도 그거야
aaa = db.similarity_search( 
    "삼성전자 사업전망에 대해 알려줘",
    k=2, # 몇 개를 가져 올건가 / 디폴트 4
)

print(aaa)
# Document > 청킹 하나, page_content=내용
