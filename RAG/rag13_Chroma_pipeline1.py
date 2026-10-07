import os, _llm


# 임베딩
DB_PAHT = "./_db/Chroma12/"
vector_store = _llm.chroma_load(DB_PAHT, collection_name="choma12")

query = "이대로 있어도 되는가?"
result = vector_store.similarity_search(query) # 검색 / 리턴 디폴트 4개

############################# Retrievers #############################
############################# 검색기 #############################
retriever = vector_store.as_retriever(search_kwargs={"k":2})


aaa =  retriever.invoke(query) # invoke(소리치다) 
print(f"검색된 관련 문서 수 : {len(aaa)}") # 2
print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}") # 앞에서 50자

print("========================================================")
############################# 모델 연결 #############################

model = _llm.getOpenAi()
response = model.invoke("AI시대에 백엔드 개발자로서 가장 중요한건 무엇인가?")
print("model의 답변 : ", response.content)
print("========================================================")
print("========================================================")

# aaa[0] 질문에 답변을 하면서 query 가 들어감
# 질문을 여러개를 하면 그 질문과 답을 모두 가지고 있다가 다시 ai에게 질문할때 그거까지 합쳐서 다시 질문한다
# query_with_context = f"""
#     {aaa[0].page_content}\n\n
#     위 내용에 대해서 근거하여 다음 질문에 답변하세요 . \n\n{query}
# """

query_with_context = f"""
    {response.content}\n\n
    위 내용에 대해서 근거하여 다음 질문에 답변하세요 . \n\n{query}
"""

# invoke 는 결국에는 predict 하는거와 같다
response = model.invoke(query_with_context)
print("model의 응답 : ", response.content)




