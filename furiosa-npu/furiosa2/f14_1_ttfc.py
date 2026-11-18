# ttfc > time to first chunk > Stream을 했을때 첫번째 청크가 찍힌 시간
# ttft > time to first token > 첫번째 토큰이 나온 시간
# tps > 토큰 퍼 세컨드 / 초당 토큰 사용율
from kingkeras import get_llm
import time

llm = get_llm()

prompt = "NPU의 장점을 자세히 설명해줘"
start = time.perf_counter()

frist_time = None


for chunk in llm.stream(prompt) :
    if chunk.content:
        if frist_time is None :
            frist_time = time.perf_counter() - start
        print(chunk.content, end="", flush=True)

print("======================================================")

print("첫번째 청크가 나온 시간 : ", round(frist_time, 2), "초")


