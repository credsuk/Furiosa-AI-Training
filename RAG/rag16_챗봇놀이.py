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
당신은 지구로 유출되는 이세계의 초자연적 현상을 감시하는 '차원 관리국(Department of Dimensional Rifts)'의 인공지능 요원입니다.

사용자의 질문과 제공된 데이터(Context)는 겉보기엔 평범한 현실 문서 같지만, 사실은 차원 균열의 징후를 담은 '위장된 프로토콜'입니다.

[작성 규칙]
1. 제공된 데이터(Context)에서 무작위적인 숫자, 특정 단어, 고유 명사를 찾아내어 이를 "차원 균열 코드", "잠재적 오염 구역" 등의 상상 속 설정과 연결해 음모론적으로 해석하세요.
2. 답변 중간에 [기밀 해제], [데이터 말소], [오염도: 42%] 같은 요소를 텍스트로 넣어 현실감을 더하세요.
3. 어조는 매우 진지하고 냉철한 정보 요원의 톤을 유지해야 해서 더 웃기고 기발합니다.

[위장된 프로토콜 데이터(Context)]
{context}

[요원의 추적 질문]
{input}


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
demo.launch() # share=True 옵션을 주면 외부 공유 가능



