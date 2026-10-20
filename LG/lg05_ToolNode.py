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

# 검색창에서 검색하는것과 동일하다
# result = web_search.invoke({
    # "query" :  "심성전자의 오늘 주가"
# })

# print(result)
# print(result['results'][-1]['content'])


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

# ToolNode는 tools를 노드로 만든다
# tool_node = ToolNode(tools)

# LLM이 알아서 판단한다
# grpah_builder.add_node("tools", tool_node)
grpah_builder.add_node("tools", ToolNode(tools))



###################### 4. 엣지 추가 #################
grpah_builder.add_edge(START, 'chatbot')
grpah_builder.add_conditional_edges( # 조건부 엣지
    'chatbot', # 할지 말지 판단
    tools_condition, # 조건 
    # tool_calls가 있으면 'tools' 반환
    # tool_calls가 없으면 END 반환
    # tools_condition에 END 포함되어 있다
)
grpah_builder.add_edge('tools', 'chatbot')
# tools에서 자동 END라고 판단되서 따로 END를 엣지 시키지 않는다

##################### 5. 그래프 컴파일 ##############
graph = grpah_builder.compile()


###################### 6. 그래프 실행 #####################
query = "심성전자의 오늘 주가는 얼마입니까?"

response = graph.invoke({
    "messages" : [("user", query)]
})


###################### 7. 결과 출력 ####################
print("AI : ", response["messages"][-1].content)

# for i, msg in enumerate(response["messages"]):
    # print(f"message {i} : {msg.content}" ) 







