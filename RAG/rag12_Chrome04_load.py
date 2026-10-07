import os, _llm


#1. 데이터
DB_PAHT = "./_db/Chroma12/"
vector_store = _llm.chroma_load(DB_PAHT, collection_name="choma12")

print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}", ) #  59

query = "삼성전자의 창업자는 누구인가요?"
result = vector_store.similarity_search(query) # 검색 / 리턴 디폴트 4개

print(f"검색 결과의 길이 : {len(result)}") # 4


######################## Retrievers ######################### 
####################### 검색기 ########################
# 백터 스코어 검색
# 검색기, 증강생성, LCL문법

retriever = vector_store.as_retriever(search_kwargs={"k":2})
# print(retriever) # 이대로 사용할 수 없음
# tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x0000023EF3B4B390> search_kwargs={'k': 2}


# 인보크 
# retriever에서 query의 유사도가 높은 걸로 반환해준다
aaa =  retriever.invoke(query) # invoke(소리치다) 

print(f"검색된 관련 문서 수 : {len(aaa)}") # 2
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}") # 앞에서 50자

