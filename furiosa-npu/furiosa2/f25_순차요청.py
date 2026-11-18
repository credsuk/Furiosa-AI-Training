import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

# 총 8개의 문장이 들어간다 / 8번을 for문을 돌린다
prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘"
    for i in range(8)
]

start = time.perf_counter()
for prompt in prompts:
    llm.invoke(prompt)
    
elapsed = time.perf_counter() - start
print("전체 요청 :", len(prompts))
print("전체 소요 시간 :", round(elapsed, 3), "초")
print("요청 처리량 : ", round(len(prompts) / elapsed, 3), "req/s") # 초당 요청량

# 전체 요청 : 8
# 전체 소요 시간 : 8.832 초
# 요청 처리량 :  0.906 req/s