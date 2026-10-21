from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 돌아가는 모양형태를 병합하는 형태를 만들기
# 그래서 선언은 간단히 한다

#1. State 정의
class State(TypedDict):
    question : str
    search_result : str
    analysis_result : str
    final_result : str

#2. 검색 노드
def search_agent(state : State):
    print("검색 에이전트 실행")
    return {"search_result" : "엔디비아는 AI GPU 시장의 주요 기업니다."}

#3. 분석 노드
def analysis_agent(state : State):
    print("분석 에이전트 실행")
    return {"analysis_result" : "AI 반도체 수요 증가가 성장 요인이다."}

#4. 결과 통합 머지(Merge) 노드
def merge_agent(state :State):
    print("병합 에이전트 실행")
    result = ( # 단순히 합쳐 주는거다
        state['search_result'] + "\n" +
        state['analysis_result'] 
    )
    return {"final_result" : result}

#5. 그래프
builder = StateGraph(State)

builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)


################# 병렬 실행 ################
# 구조를 만들어 주면 된다
builder.add_edge(START, "search")
builder.add_edge(START, "analysis")

################# 병합 #####################
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")

builder.add_edge("merge", END)


# 컴파일
graph = builder.compile()

# 실행
response = graph.invoke({
    "questring" : "엔디비아의 미래 전망은?"
})

print(response['final_result'])

###################### 그래프 이미지 ######################
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12, 8)) # 통상 인치 / 그림판 사이즈
plt.imshow(img)
plt.axis('off') # 축이 보이지 않음
plt.show()



