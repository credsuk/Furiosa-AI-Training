# RPS : Request Per Second

import time
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=100)

# 총 8개의 문장이 들어간다 / 8번을 for문을 돌린다
prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘"
    for i in range(16)
]


def ask(prompt):
    response = llm.invoke(prompt)
    usage = response.usage_metadata or {}
    return usage.get("output_tokens", 0)

start = time.perf_counter()

with ThreadPoolExecutor(max_workers=4) as executor:
    tokens = list(executor.map(ask, prompts))
elapsed = time.perf_counter() - start


total_tokens = sum(tokens)

print("전체 출력 토큰 : ", total_tokens)
print("전체 시간 :", round(elapsed, 2), "초")
print("Output TPS : ", round(total_tokens / elapsed, 2))
print("RPS : ", round(len(prompts) / elapsed, 2))

# 전체 출력 토큰 :  1600
# 전체 시간 : 4.9 초
# Output TPS :  326.58
# RPS :  3.27