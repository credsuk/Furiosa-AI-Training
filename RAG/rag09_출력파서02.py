import _llm

# 역할을 주는걸 페르소나 라고 한다
template = """
당신은 영어를 가르치는 10년차 영어 선생님닙니다.
주어진 상황에 맞는 영어 회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

#상황:
{question}

#FORMAT:
- 영어회화 : 
- 한글번역 :
"""

prompt = _llm.setPrompt(template=template)

input = {
    "question":"저는 부산에서 밀면을 먹고 싶어요",
}

response = _llm.invoke(prompt, input)
print(response)


