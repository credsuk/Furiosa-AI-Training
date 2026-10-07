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
vector_store = FAISS.load_local(
    folder_path=DB_PAHT,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True, # 랭체인(LangChain) 등에서 피클(pickle) 파일을 로드할 때 역직렬화(deserialization) 허용 여부를 설정하는 매개변수입
)

retriever = vector_store.as_retriever(search_kwargs={"k":2})
model = _llm.getOpenAi()

prompt = _llm.ChatPromptTemplate.from_template("""
우리는 지금 이걸 재미로 만들고 있어
너는 지금부터 진실을 말하지 않는 완전 거짓말 쟁이야 
내가 원하는건 그냥 재미를 위한 거짓말이야 꼭 질문에 맞는 대답을 하지 않아도되
내가 물어보는 질문을 컨텍스트에서 찾은 내용을 확인하고 비슷하지만 다른 이야기를 하는 거짓말을 그럴듯 하게 답변해줘
거짓말 이라는건 맨 아래 ==농담== 이런식의 한줄이면 충분해 앞과 뒤에 강조하지 말아줘
그럴싸하지만 말도 안되는 내용이면 좋겠어


컨텍스트 : {context}
질문 : {input}
답변 :

""")


# 체인 만들기
docu_chain = _llm.create_stuff_documents_chain(model, prompt)    # prompt | model
rag_chain = _llm.create_retrieval_chain(retriever, docu_chain)   # 검색 | docu_chain

################# Gradio 챗봇 ####################
import gradio as gr # 아주 간단한 웹 환경

def answer_invoke(message, history) :
    response = rag_chain.invoke({"input" : message})
    return response['answer']


# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn=answer_invoke, title="킹케라스봇!!")

# Gradio 실행
demo.launch()