from kingkeras import get_llm
from langchain_core.tools import tool # tools는 도구

#1. 도구 정의
@tool # @이는 데코레이터 
def add(aaa: int, bbb : int) -> int: # 타입힌트
    """ 두 정수를 더합니다."""
    return aaa + bbb

#2. llm
llm = get_llm()

#3.도구 등록
# tools colling
llm_with_tools = llm.bind_tools([add])

#4. 질문
response = llm_with_tools.invoke("3202과 131을 add로 더해줘.")

#5. 결과 확인
for call in response.tool_calls:
    result = add.invoke(call['args'])
    print("계산 결과 : ", result)
