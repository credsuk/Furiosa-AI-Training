from langchain_openai import ChatOpenAI
import os # OS에 키를 자동으로 불러오는 기능이 존재한다
# 내 컴퓨터 환경변수에 그대로 넣겠다 라는 뜻
os.environ["OPENAI_API_KEY"] = "-9roACfGTXucIUSBIOPjfTBO5be_hKecA"

llm  = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, #
    # api_key= openai_api_key,
)

response = llm.invoke("안녕하세요.")
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?




