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
    loader = TextLoader(text_file,  encoding='utf-8')
    data += loader.load()

# print(len(data)) # 3 개를 넣었으니
# print(data[0].page_content)

# TextLoader로 불러온 3개의 데이터로 되어 있다
# 데이터를 불러와서 doc에 넣고 doc.page_content에 길이를 넣는다
# 간략화된 for문
char_count = [len(doc.page_content) for doc in data]
print(char_count) # [8158, 2049, 1898]


#02. 문서를 자른다 / 청킹
text_splitter = _llm.getTextSplitter()

texts = text_splitter.split_documents(data)

# print("생성된 텍스트 청크수 : ", len(texts)) # 생성된 텍스트 청크수 :  59 > 벡터DBd에 59개가 생길 예정
# print("각 청크의 길이 : ", list(len(text.page_content) for text in texts))
# 청크의 길이는 chunk_size를 넘지 않는 사이즈에서 separators를 기준으로 자른값이 나오기 때문에 chunk_size가 300d이라면 300개 언너지가 나온다

# print("첫번째 청크의 내용 : ", texts[0]) # page_content = 내용, metadata = 경로다
# print("첫번째 청크의 길이 : ", len(texts[0].page_content)) # 259


#03. 임베딩
# 우리가 차원이라고 이야기하면서 스칼라다
DB_PAHT = "./_db/Chroma12/"

# 저장
# vector_store = Chroma.from_documents(
#     documents=texts, # 문서
#     embedding=_llm.getEmbedding(),  # OpenAi Embedding
#     persist_directory=DB_PAHT, # 경로
#     collection_name='croma12', # 이름
# )

vector_store = _llm.chroma_save(texts, persist_directory = DB_PAHT, collection_name='choma12')

print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}", ) #  59
# 청크 59개의 수와 같이 저장된다

query = "삼성전자의 창업자는 누구인가요?"
result = vector_store.similarity_search(query) # 검색 / 리턴 디폴트 4개

print(f"검색 결과의 길이 : {len(result)}") # 4


######################## Retrievers ######################### 
####################### 검색기 ########################
# 백터 스코어 검색
# 검색기, 증강생성, LCL문법

retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever) # 이대로 사용할 수 없음
# tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x0000023EF3B4B390> search_kwargs={'k': 2}


# 인보크 
# retriever에서 query의 유사도가 높은 걸로 반환해준다
aaa =  retriever.invoke(query) # invoke(소리치다) 

print(f"검색된 관련 문서 수 : {len(aaa)}") # 2
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}") # 앞에서 50자



