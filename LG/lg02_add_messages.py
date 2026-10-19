from typing import Annotated, TypedDict
from langgraph.graph import add_messages


# Type > 형태, Dict > 딕셔너리 합쳐서 형태 문서
# (TypedDict)은 class에서는 상속 받겠다 / 함수에서는 인자로 받는거
class MyDate(TypedDict):
    messages : Annotated[list, add_messages]  # Reducer / 지금은 add_messages 가 생각하게 하자
# Annotated > 주석
# 값들은 앞으로 리스트형태로 있을꺼다 그걸 이어 붙여주는게 add_messages
# list에다가 자동 append를 하는 함수
# 이거 자주쓰게 될꺼야

from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content='안녕', id='1')]
msg2 = [AIMessage(content='방가', id='2')]

results = add_messages(msg1, msg2)
# print(results)
# [HumanMessage(content='안녕', additional_kwargs={}, response_metadata={}, id='1'), AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='2', tool_calls=[], invalid_tool_calls=[])]


#############################
msg3 = [HumanMessage(content='안녕', id='3')]
msg4 = [AIMessage(content='방가', id='3')]

results2 = add_messages(msg3, msg4)
print(results2)
# [AIMessage(content='방가', additional_kwargs={}, response_metadata={}, id='3', tool_calls=[], invalid_tool_calls=[])]
# ID가 동일하면 덮어써진다






