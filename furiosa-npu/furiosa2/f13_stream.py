from kingkeras import get_llm

llm = get_llm(max_tokens=2000)

prompt = "인공지능의 발전 과정을 설명해줘."

for chunk in llm.stream(prompt) :
    print(chunk.content, end="", flush=True), # end=줄바꿈 하지 마세요 / flush = 덮어 씌워라
