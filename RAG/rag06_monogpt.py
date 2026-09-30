# from langchain_openai import ChatOpenAI
# import os
# from dotenv import load_dotenv
# load_dotenv()

# api_key = os.environ["MONOROUTER_API_KEY"].strip() # 줄바꿈 띄어씌기 인정 안함
# base_url = os.environ["MONOROUTER_BASE_URL"].strip()

# llm = ChatOpenAI(
#     model_name='gpt-5.6-terra',
#     temperature=0, #
#     api_key= api_key,
#     base_url=base_url, # 보통 외부키는 이렇게 해야 연결 가능
# )

# response = llm.invoke("안녕하세요.")
# print(response.content) # 안녕하세요! 무엇을 도와드릴까요?



## 나 혼자 UTIL로 간소화
import _llm as _llm
llm = _llm.getOpenAi()
response = llm.invoke("안녕하세요.")
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?