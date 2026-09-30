# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser
# parser :  분석하다. 답벼을 다듬다.

import _llm
from langchain_core.output_parsers import StrOutputParser # 출력을 정리해줌

prompt = _llm.setPrompt("{topic}에 대해 쉽게 설명해주세요.")
input = {"topic":"양자컴퓨터 학습 원리"}

response = _llm.invoke(prompt, input)
print(response)
