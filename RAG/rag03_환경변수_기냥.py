from langchain_openai import ChatOpenAI
import os 


llm  = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, #
    # api_key= openai_api_key,
)

response = llm.invoke("안녕하세요.")
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?




