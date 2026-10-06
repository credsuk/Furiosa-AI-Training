# 계속 쓰기 귀찮아 내가 만든 클래스
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv # env파일을 읽어오는 플러그인
load_dotenv() # env파일 불러오기

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 줄바꿈 띄어씌기 인정 안함
base_url = os.environ["MONOROUTER_BASE_URL"].strip()

# ChatOpenAI Model 불러오기
def getOpenAi():
    return ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, #
    api_key= api_key,
    base_url=base_url, # 보통 외부키는 이렇게 해야 연결 가능
)

# PromptTemplate 불러오기
def setPrompt(*, str=None, template=None):
    if template:
        return PromptTemplate.from_template(template=template)
    elif str:
        return PromptTemplate.from_template(str)
    else:
        raise ValueError("str 또는 template 중 하나는 지정해야 합니다.")

def getParser(*, type="str"):
    if type=="str" | type==None :
        return StrOutputParser()


def invoke(prompt, input, parser=None):
     # parser가 없거나 비어있다면 기본값 지정
    if not parser:
        parser = getParser()

    chain = prompt | getOpenAi() | parser
    return chain.invoke(input)


def chroma_save(texts, *, persist_directory="./_db/Chroma/", collection_name="croma"): 
    return Chroma.from_documents(
        documents=texts, # 문서
        embedding=getEmbedding(),  # OpenAi Embedding
        persist_directory=persist_directory, # 경로
        collection_name=collection_name, # 이름
    )


def getEmbedding() :
    return OpenAIEmbeddings(
        model="text-embedding-3-small", # 1,536차원 / 임베딩을 해주는 모델 끝날때까지 이거쓰면 충분
        # model="text-embedding-3-large", # 3,072차원
        api_key=api_key,
        base_url=base_url,
    )

def getEmbeddingQuery(prompt) :
    embeddings = getEmbedding()
    return embeddings.embed_query(prompt) # 한줄 일때는 embed_query
    

def getTextSplitter(*, splitter=None, chunk_size=300, chunk_overlap=100) :
    textSplitter = ""
    if splitter == 'TextSplitter' :
        textSplitter = TextSplitter(
            chunk_size=chunk_size, # 자르는 사이즈
            chunk_overlap=chunk_overlap, # 중복되는 구간,  끝나는 지점을 중복 시킨다
            separators=["\n\n", "\n", " ", ""],  # 통상 디폴트, / 문단바꿈, 줄바꿈, 빈공간,  | 재귀 함수
        )
    elif splitter == 'CharacterTextSplitter' :
        textSplitter = CharacterTextSplitter(
            chunk_size=chunk_size, # 자르는 사이즈
            chunk_overlap=chunk_overlap, # 중복되는 구간,  끝나는 지점을 중복 시킨다
            separators=["\n\n", "\n", " ", ""], 
        )
    else :
        textSplitter = RecursiveCharacterTextSplitter( # 재귀적인 텍스트 스플리터
            chunk_size=chunk_size, # 자르는 사이즈
            chunk_overlap=chunk_overlap, # 중복되는 구간,  끝나는 지점을 중복 시킨다
            separators=["\n\n", "\n", " ", ""], 
        )

    return textSplitter