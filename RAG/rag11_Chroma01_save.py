import os, _llm
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma


# 데이터를 불러온다 > 데이터를 청큰(자른다)한다 > 벡터화 시켜서 DB에 넣는다

#01. 데이터 불러오기
textPath = "./_data/rag_data/"
loader1 = TextLoader(file_path=textPath + "samsung_outlook.txt", encoding='utf-8')
loader2 = TextLoader(file_path=textPath + "nvidia_outlook.txt",  encoding='utf-8')
# loader3 = TextLoader(file_path=textPath + "2026_AI_for_All.txt", )


#02. 문서를 자른다 / 청킹
# 자르는 기준을 먼저 정의 
# 보통 RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter 이거 많이 씀
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # 자르는 사이즈
    chunk_overlap=100, # 중복되는 구간,  끝나는 지점을 중복 시킨다
    separators=["\n\n", "\n", " ", ""],  # 통상 디폴트,
)

# _llm.py에서 TextSplitter 불러오기
# text_splitter = _llm.getTextSplitter()

split_doc1 = loader1.load_and_split(text_splitter=text_splitter)
split_doc2 = loader2.load_and_split(text_splitter=text_splitter)

print(split_doc1) # 청크 300, 오버랩 100 으로 자른다
print(len(split_doc1), len(split_doc2)) # 9 9


# 우리가 차원이라고 이야기하면서 스칼라다
DB_PAHT = "./_db/Chroma11/"

# 저장
db = Chroma.from_documents(
    documents=split_doc1 + split_doc2, # 문서
    embedding=_llm.getEmbedding(),  # OpenAi Embedding
    persist_directory=DB_PAHT, # 경로
    collection_name='croma11', # 이름
)

print("Chroma 문서서장 끝")




