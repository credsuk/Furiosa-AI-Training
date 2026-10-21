# tool calling : 외부 기능을 가져와서 사용하기!!

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip() # 줄바꿈 띄어씌기 인정 안함
base_url = os.environ["MONOROUTER_BASE_URL"].strip()
tavily_api_key = os.environ["TAVILY_API_KEY"].strip()

llm = ChatOpenAI(
    model_name='gpt-6-luna',
    temperature=0,
    max_tokens=1000,
    api_key= api_key,
    base_url=base_url, # 보통 외부키는 이렇게 해야 연결 가능
)


###################################### 0. Tool 정의 ##########################################
# pip install langchain_tavily
from langchain_tavily import TavilySearch

# 웹 검속 도구 생성
web_search =  TavilySearch(
    max_results=3,
)


tools = [web_search]

llm.bind_tools(tools=tools) # llm에 web search 함친거
llm_with_tools = llm.bind_tools(tools=tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages # Reducer 
from langgraph.prebuilt import ToolNode, tools_condition


##################### 1. State 정의 #####################
class State(TypedDict) :
    messages : Annotated[list, add_messages]


#################### 2. node 정의 #####################
def chatbot(state : State) : 
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}


################### 3. 그래프 구성, 노드 추가 #################
# 그래프 생성
grpah_builder = StateGraph(State)

grpah_builder.add_node("chatbot", chatbot)

grpah_builder.add_node("tools", ToolNode(tools))



###################### 4. 엣지 추가 #################
grpah_builder.add_edge(START, 'chatbot')
grpah_builder.add_conditional_edges( # 조건부 엣지
    'chatbot', # 할지 말지 판단
    tools_condition, # 조건 
)
grpah_builder.add_edge('tools', 'chatbot')

##################### 5. 그래프 컴파일 ##############
graph = grpah_builder.compile()


###################### 6. 그래프 실행 #####################

# 첫번째 질문
response = graph.invoke({
    "messages" : [("user", "내 이름은 김기관이야.")]
})
print(response["messages"][-1].content)

print("================== 두번째 질문 ======================")

# 두번째 질문
response = graph.invoke({
    "messages" : [("user", "내 이름이 뭐라고 했지?")]
})
print(response["messages"][-1].content)

# 이렇게 했을데 연결이 되는지 확인하려고 한다






