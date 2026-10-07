import os, _llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# 임베딩
DB_PAHT = "./_db/Chroma12/"

vector_store = _llm.chroma_load(DB_PAHT, collection_name="choma12")
############################# Retrievers #############################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
model = _llm.getOpenAi()

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로 답변할 수 없습니다." 라고 말씀해 주세요.

컨텍스트 : {context}
질문 : {input}
답변 :

""")


# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)    # prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain)   # 검색 | docu_chain

"""
# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input" : query})

print(response)
print("========================================================")
print(response.keys()) # dict_keys(['input', 'context', 'answer'])
print("========================================================")
print(response['answer']) # 주어진 정보로 답변할 수 없습니다.
"""


################# Gradio 챗봇 ####################
################# Gradio 챗봇 ####################
import gradio as gr # 아주 간단한 웹 환경

def answer_invoke(message, history) :
    response = rag_chain.invoke({"input" : message})
    return response['answer']


# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn=answer_invoke, title="킹케라스봇!!")

# Gradio 실행
demo.launch()



