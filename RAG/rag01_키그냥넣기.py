from langchain_openai import ChatOpenAI

openai_api_key = "9roACfGTXucIUSBIOPjfTBO5be_hKecA"

llm  = ChatOpenAI(
    model_name='gpt-5.6-terra', # 모델 선택
    temperature=0, #
    api_key= openai_api_key, # api key 세팅
)

response = llm.invoke("안녕하세요.") # invoke소리치다 라는 뜻
print(response.content) # 안녕하세요! 무엇을 도와드릴까요?

# 현재는 메모리 기능이 없어서 그냥 휘발성으로 날라간다
# 이전 프롬프트를 다시 넣는게 메모리 기능, 그래서 돈이 많이 나감



