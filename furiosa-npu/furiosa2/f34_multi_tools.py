from kingkeras import get_llm
from langchain_core.tools import tool # tools는 도구

#1. 함수들 정의
@tool 
def add(aaa: int, bbb : int) -> int: # 타입힌트
    """ 두 정수를 더합니다."""
    return aaa + bbb

@tool
def multiply(a: int, b: int) -> int :
    """ 두 정수를 곱합니다."""
    return a * b


@tool
def subtract(a: int, b: int) -> int :
    """ 두 정수를 뺍니다."""
    return a - b


#2. 도구 목록
tools = [add, multiply, subtract]

tool_map = {tool.name : tool for tool in tools}
# print("=============================")
# print(tool_map)
# print(tool_map['add'])
# name='add' description='두 정수를 더합니다.' args_schema=<class 'langchain_core.utils.pydantic.add'> func=<function add at 0x7ef912038b80>

#3. llm + 도구 연결
llm = get_llm(max_tokens=300)
llm_with_tools = llm.bind_tools(tools)


#4. 프롬프트
questions = [
    "15와 27일 더해줘",
    "12와 8을 곱해줘",
    "200에서 150을 빼줘"
]

for question in questions :
    response = llm_with_tools.invoke(question)
    print("\n질문 : ", question)

    for call in response.tool_calls:
        print("선택한 도구 :", call['name'])
        print("매개 변수 :", call['args'])
        result = tool_map[call['name']].invoke(call['args'])
        print("실행 결과 : ", result)

