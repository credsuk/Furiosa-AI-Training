import os, _llm
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma


# 폴더에서 텍스트 파일 목록 가져오기
from glob import glob # 경로를 땡겨줌
textPath = "./_data/rag_data/"
txt_files = glob(os.path.join(textPath, "*.txt")) # 해당경로에 있는 모든 .txt파일을 리스트 형태로 가져와라

print(txt_files) # ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']


#01. 데이터 불러오기
data = []
for text_file in txt_files:
    loader = TextLoader(file_path=text_file,  encoding='utf-8')
    data.append(loader)

print(data)

exit()

#02. 문서를 자른다 / 청킹
text_splitter = _llm.getTextSplitter()

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




