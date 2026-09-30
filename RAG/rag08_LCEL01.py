# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 줄바꿈 띄어씌기 인정 안함
base_url = os.environ["MONOROUTER_BASE_URL"].strip()

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해주세요.")

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, #
    api_key= api_key,
    base_url=base_url, # 보통 외부키는 이렇게 해야 연결 가능
)

# 오버라이딩 > 다시 재정의 했다 
# 이게 왜 오버라이딩이지?;;
chain = prompt | model  # prompt 와 model은 연결관계로 지정하겠다

input = {"topic":"양자컴퓨터 학습 원리"}

# input 들어가서 prompt를 연결하고 model로연결해라
response = chain.invoke(input)

print(response.content)

