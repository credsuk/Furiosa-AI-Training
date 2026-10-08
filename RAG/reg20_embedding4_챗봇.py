# reg20_embedding2_bge-m3 복사

import _llm
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



################################## 데이터 저장 ##############################
# #01. 데이터 불러오기
# path = "./_data/"
# pdf_loader = PyPDFLoader(path + "Attention Is All You Need.pdf")
# pdf_docs = pdf_loader.load()
# print("데이터 불러오기")

# #02. 문서를 자른다 / 청킹
# text_splitter = _llm.getTextSplitter()
# split_doc = pdf_loader.load_and_split(text_splitter=text_splitter)
# print("청킹")

# #03. 임베딩
# embeddings = HuggingFaceEmbeddings(
#     # 베이징 어느 대학교에서 나온건데 말귀를 잘 알아 듣는 모델이다 (한국, 중국, 일본은 아주 좋음)
#     model_name="BAAI/bge-m3", 
#     model_kwargs={
#         "device" : "cpu", # GPU : cuda / CPU : cpu
#         "local_files_only" : True, # 다운로드 받지말고 먼저 받아둔 이미지를 사용해라
#     }
# )

# # embeddings = HuggingFaceEmbeddings(
# #     # 베이징 어느 대학교에서 나온건데 말귀를 잘 알아 듣는 모델이다 (한국, 중국, 일본은 아주 좋음)
# #     model_name="Qwen/Qwen3-Embedding-0.6B", 
# #     model_kwargs={
# #         "device" : "cpu", # GPU : cuda / CPU : cpu
# #         # "local_files_only" : True,
# #     },
# # )

# print("임베딩")

# db = FAISS.from_documents(
#     documents=split_doc,
#     embedding=embeddings, 
# )
# print("from_documents")

# DB_PAHT = "./_db/rag20/"
# db.save_local(
#     folder_path=DB_PAHT, # persist_directory 동일
#     index_name='rag20', # collection_name 동일
# )
# print("DB 저장 완료")



# exit()

################################## 데이터 불러오기 ################################
#01. 데이터 불러오기
embeddings = HuggingFaceEmbeddings(
    # 베이징 어느 대학교에서 나온건데 말귀를 잘 알아 듣는 모델이다 (한국, 중국, 일본은 아주 좋음)
    model_name="BAAI/bge-m3", 
    model_kwargs={
        "device" : "cpu", # GPU : cuda / CPU : cpu
        "local_files_only" : True, # 다운로드 받지말고 먼저 받아둔 이미지를 사용해라
    }
)

# embeddings = HuggingFaceEmbeddings(
#     # 베이징 어느 대학교에서 나온건데 말귀를 잘 알아 듣는 모델이다 (한국, 중국, 일본은 아주 좋음)
#     model_name="Qwen/Qwen3-Embedding-0.6B", 
#     model_kwargs={
#         "device" : "cpu", # GPU : cuda / CPU : cpu
#         # "local_files_only" : True,
#     },
# )

DB_PAHT = "./_db/rag20/"
vector_store = FAISS.load_local(
    folder_path=DB_PAHT,
    embeddings=embeddings,
    index_name='rag20',
    allow_dangerous_deserialization=True, 
)

# 검색 조건 세팅
retriever = vector_store.as_retriever(search_kwargs={"k":2})


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



