# 챗봇
# transformer 논문을 벡터디비로 불러와서
# 요약, 인용 등등 할수 있는 챗봇으로!!

import os, _llm
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import faiss 
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


################################## 데이터 저장 ##############################
# #01. 데이터 불러오기
# path = "./_data/"
# pdf_loader = PyPDFLoader(path + "Attention Is All You Need.pdf")
# pdf_docs = pdf_loader.load()


# #02. 문서를 자른다 / 청킹
# text_splitter = _llm.getTextSplitter()
# split_doc = pdf_loader.load_and_split(text_splitter=text_splitter)

# #03. 임베딩
# embeddings = _llm.getEmbedding()

# db = FAISS.from_documents(
#     documents=split_doc,
#     embedding=embeddings,  # OpenAi Embedding
# )

# #04. 저장
# DB_PAHT = "./_db/rag19/"
# db.save_local(
#     folder_path=DB_PAHT, # persist_directory 동일
#     index_name='rag19', # collection_name 동일
# )


################################## 데이터 불러오기 ################################
#01. 데이터 불러오기
embeddings = _llm.getEmbedding()

DB_PAHT = "./_db/rag19/"
vector_store = FAISS.load_local(
    folder_path=DB_PAHT,
    embeddings=embeddings,
    index_name='rag19',
    allow_dangerous_deserialization=True, # 랭체인(LangChain) 등에서 피클(pickle) 파일을 로드할 때 역직렬화(deserialization) 허용 여부를 설정하는 매개변수입
)

# 검색 조건 세팅
retriever = vector_store.as_retriever(search_kwargs={"k":3})


#02. AI질문 사전 세팅
# 판단할 OpenAi 연동
model = _llm.getOpenAi()

# 프롬프트 작성
prompt = _llm.ChatPromptTemplate.from_template("""
백터DB에서 읽은 내용을 바탕으로 초등학생도 이해할 수 있는 단어와 예시를 들어서 해섞해주세요
거짓말은 하지말고 혹시 모르는게 있다면 추가로 검색을 해주세요
한글로 질문한 검색을 영어로 변경하고, 영어로 검색된 내용은 한글로 풀어서 이야기해주세요 즉, 사용자와의 대화는 한글을 이용해주세요
사용자에게 확인해야할 사항이 있다면 확인을 하고 진행해주세요


[Context]
{context}

[사용자 질문]
{input}

""")


#03. 체인 만들기
docu_chain = _llm.create_stuff_documents_chain(model, prompt)    # prompt | model
rag_chain = _llm.create_retrieval_chain(retriever, docu_chain)   # 검색 | docu_chain


#04. 챗봇 연동
import gradio as gr # 아주 간단한 웹 환경

def answer_invoke(message, history) :
    response = rag_chain.invoke({"input" : message})
    return response['answer']


# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn=answer_invoke, title="논문 검색")

# Gradio 실행
demo.launch() # share=True 옵션을 주면 외부 공유 가능


