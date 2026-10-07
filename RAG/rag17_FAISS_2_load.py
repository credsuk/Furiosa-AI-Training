import os, _llm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma


import faiss 
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


# 데이터를 불러온다 > 데이터를 청큰(자른다)한다 > 벡터화 시켜서 DB에 넣는다

#03. 임베딩
embeddings = _llm.getEmbedding()

DB_PAHT = "./_db/Faiss17/"
db = FAISS.load_local(
    folder_path=DB_PAHT,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True, # 랭체인(LangChain) 등에서 피클(pickle) 파일을 로드할 때 역직렬화(deserialization) 허용 여부를 설정하는 매개변수입
)

print("============================================================")

# 문서 저장소 ID확인
print(db.index_to_docstore_id)
print("============================================================")

# 저장된 결과 확인
print(db.docstore.__dict__)
print("============================================================")

# 유사도 검색
aaa = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2)
print(aaa)




