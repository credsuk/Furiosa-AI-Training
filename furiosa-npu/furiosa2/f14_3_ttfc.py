from kingkeras import get_llm
import time

llm = get_llm()

prompt = "NPU의 장점을 자세히 설명해줘"
start = time.perf_counter()


chunk = next(iter(llm.stream(prompt)))

frist_time = time.perf_counter() - start

print("======================================================")
print("첫번째 청크가 나온 시간 : ", round(frist_time, 2), "초")


