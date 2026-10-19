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
from langchain_core.messages import AIMessage, HumanMessage


#################### 1. State 정의 #####################
class State(TypedDict) :
    messages : Annotated[list, add_messages]

###################### 2. node 정의 ######################
def prepare(state : State) :
    return {"messages" : [AIMessage(content="질문을 접수했습니다.")]}

def chatbot(state : State) : 
    return {"messages" : [llm.invoke(state["messages"])]}

def finish(state : State) :
    return {"messages" : [AIMessage(content="답변 처리가 완료되었습니다.")]}

###################### 3. 그래프 구성 #######################
graph_01 = StateGraph(State) # Sequential 해당 모델 느낌으로 기억하면 좋다
graph_01.add_node("prepare", prepare) # 이름은 달라도 되지만 햇갈리니깐 동일하게 작성해주자
graph_01.add_node("chatbot", chatbot) 
graph_01.add_node("finish", finish) 


####################### 4. 엣지 추가 ###########################
graph_01.add_edge(START, "prepare") 
graph_01.add_edge("prepare", "chatbot")
graph_01.add_edge("chatbot", "finish")
graph_01.add_edge("finish", END) 


######################## 5. 그래프 컴파일 ########################
graph = graph_01.compile()


####################### 6. 그래프 실행 ########################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
    "messages" : [("user", query)]
})


###################### 7. 결과 출력 ####################
# enumerate 함수 확인해봐야함
for i, msg in enumerate(response["messages"]):
    print(f"message {i} : {msg.content}" ) 







