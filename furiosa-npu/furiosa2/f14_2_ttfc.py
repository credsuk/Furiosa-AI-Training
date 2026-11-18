from kingkeras import get_llm
import time

llm = get_llm()

prompt = "NPU의 장점을 자세히 설명해줘"
start = time.perf_counter()

frist_time = None


for chunk in llm.stream(prompt) :
    if chunk.content :
        frist_time = time.perf_counter() - start
        break

print("======================================================")
print("첫번째 청크가 나온 시간 : ", round(frist_time, 2), "초")


