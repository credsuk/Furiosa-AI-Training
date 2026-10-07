import os, _llm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma


import faiss 
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


# 데이터를 불러온다 > 데이터를 청큰(자른다)한다 > 벡터화 시켜서 DB에 넣는다

#01. 데이터 불러오기
textPath = "./_data/rag_data/"
loader1 = TextLoader(file_path=textPath + "samsung_outlook.txt", encoding='utf-8')
loader2 = TextLoader(file_path=textPath + "nvidia_outlook.txt",  encoding='utf-8')


#02. 문서를 자른다 / 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # 자르는 사이즈
    chunk_overlap=100, # 중복되는 구간,  끝나는 지점을 중복 시킨다
    separators=["\n\n", "\n", " ", ""],  # 통상 디폴트,
)

split_doc1 = loader1.load_and_split(text_splitter=text_splitter)
split_doc2 = loader2.load_and_split(text_splitter=text_splitter)

# print(split_doc1) # 청크 300, 오버랩 100 으로 자른다
# print(len(split_doc1), len(split_doc2)) # 9 9


#03. 임베딩
embeddings = _llm.getEmbedding()


############### 요기부터 faiss #################
faiss_index = faiss.IndexFlatL2( # 유클리드 거리로 검색을 하겠다
    len(embeddings.embed_query("Hello world")), # 몇개의 차원을 쓸꺼야 
)
faiss_index = faiss.IndexFlatL2(1536) # 그래서 이렇게도 가능
print("FAISS 인덱스 초기화 준비 완료")

# FAISS 벡터 저장소의 벡터 차원 수 (인베딩 차원 수)
print(faiss_index.d)

faiss_db = FAISS(
    embedding_function=embeddings,
    index = faiss_index,
    docstore=InMemoryDocstore(), # 메모리 기반으로 작업 하겠다
    index_to_docstore_id={}, # 인덱스에 들어가는 doc ID가 뭐냐라는건데 없으니깐 빈값
)

# 저장된 문서의 갯수 확인
print(faiss_db.index.ntotal) # 0

################ 준비 완료 ######################
################################################

db = FAISS.from_documents(
    documents=split_doc1 + split_doc2, # 문서
    embedding=embeddings,  # OpenAi Embedding
)
DB_PAHT = "./_db/Faiss17/"
db.save_local(
    folder_path=DB_PAHT, # persist_directory 동일
    index_name='faiss_index17', # collection_name 동일
)



# 크로마와 차이점을 위해 주석 처리
# db = Chroma.from_documents(
#     documents=split_doc1 + split_doc2, # 문서
#     embedding=_llm.getEmbedding(),  # OpenAi Embedding
#     persist_directory=DB_PAHT, # 경로
#     collection_name='croma11', # 이름
# )










