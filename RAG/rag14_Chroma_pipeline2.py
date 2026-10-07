import os, _llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# 임베딩
DB_PAHT = "./_db/Chroma12/"
vector_store = _llm.chroma_load(DB_PAHT, collection_name="choma12")

############################# Retrievers #############################
############################# 검색기 #############################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
model = _llm.getOpenAi()


prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로 답변할 수 없습니다." 라고 말씀해 주세요.

컨텍스트 : {context}
질문 : {input}
답변 :

""")


# 체인 만들기 / 저 랭체인이 친하다 자주 같이 쓰임
docu_chain = create_stuff_documents_chain( # 영어를 해석하면 / 만들겠다_재료_문서들_체인
    model, prompt # prompt | model 과 같다
)

rag_chain = create_retrieval_chain( # 
    retriever, docu_chain # 백터DB에서 검색하겠다 / 검색 | docu_chain > 검색 | prompt | model
)

# retriever에서 context를 생해줌
# input으로 받은 질문을 벡터DB를 검색해 두개의 청크를 컨텐스트에 알아서 포함시켜서 model로 검색한 후 데이터를 알려줌
# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input" : query})

print(response)
print("========================================================")
print(response.keys()) # dict_keys(['input', 'context', 'answer'])
print("========================================================")
print(response['answer']) # 주어진 정보로 답변할 수 없습니다.



