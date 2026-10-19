from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 줄바꿈 띄어씌기 인정 안함
base_url = os.environ["MONOROUTER_BASE_URL"].strip()

llm = ChatOpenAI(
    model_name='gpt-6-luna',
    temperature=0,
    max_tokens=1000,
    api_key= api_key,
    base_url=base_url, # 보통 외부키는 이렇게 해야 연결 가능
)

# response = llm.invoke("안녕하세요.")
# print(response.content) # 안녕하세요! 무엇을 도와드릴까요?

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages # Reducer 


# Query
# Retrieval
# Model
# Answer

#################### 1. State 정의 #####################
class State(TypedDict) :
    messages : Annotated[list, add_messages]
    aaa : str # 이렇게도 가능하다
    bbb : str 
    ccc : int

# 이렇게 작성하면 위에 스테이트 상태랑 동일하다
# state = {
#     "messages" : [],
#     "aaaa" : "abc", 
#     "bbb" : "bcc",
#     "ccc" : 123,
# }

# 위에서 스테이트를 받아서 아래로 전달해준다

###################### 2. node 정의 ######################
# 함수로 구성되어 진다
# State형태를 받아 들일꺼다
def chatbot(state : State) : 
    return {"messages" : [llm.invoke(state["messages"])]}


###################### 3. 그래프 구성 #######################
# 그래프 이렇게 만들꺼야 정도로 기억하자
graph_01 = StateGraph(State) # Sequential 해당 모델 느낌으로 기억하면 좋다
graph_01.add_node("chatbot", chatbot) # 계속 add_node할 수 있다


####################### 4. 엣지 추가 ###########################
# 실제 연결은 엣지로 연결한다
graph_01.add_edge(START, "chatbot") # 그냥 시작하라는 명령어로 생각하자
# 시작해서 챗봇 노드를 돌려라

# ("이거 들어가서", "이게 나온다")
graph_01.add_edge("chatbot", END) # 챗봇에서 노드를 끝내라

# 함수형 모델이라고 생각하면 된다
# 합쳤다가, 위로 갔다가, 내려갔다가 이것저것 할 수 있다


######################## 5. 그래프 컴파일 ########################
graph = graph_01.compile()


####################### 6. 그래프 실행 ########################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
    "messages" : [("user", query)]
})


print(response)
print(response["messages"][0].content) # human
print(response["messages"][1]) # ai
print(response["messages"][0].type) 
print(response["messages"][1].type) 






