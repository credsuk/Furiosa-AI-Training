from kingkeras import get_llm
from langchain_core.tools import tool # tools는 도구

#1. 도구 정의
@tool # @이는 데코레이터 / 기존에 만들어놓은걸 사용한다
def add(aaa: int, bbb : int) -> int: # 타입힌트
    """ 두 정수를 더합니다."""
    return aaa + bbb

# tool함수를 이용할때는 """주석""" 이 꼭 필요하다 그건 Description으로 동작한다

print(add.name) # add
print(add.description) # 두 정수를 더합니다.
print(add.args) # {'aaa': {'title': 'Aaa', 'type': 'integer'}, 'bbb': {'title': 'Bbb', 'type': 'integer'}}


#2. llm
llm = get_llm()

#3.도구 등록
# tools colling
llm_with_tools = llm.bind_tools([add])

#4. 질문
response = llm_with_tools.invoke("3232과 133을 add로 더해줘.")

#5. 결과 확인
print("일반 응답 : ", response.content)
print("====================================")
print("Tool Calls : ", response.tool_calls)
print("====================================")
print("메타 데이터 : ", response.response_metadata)

