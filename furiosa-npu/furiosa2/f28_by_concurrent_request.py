# 27 카피
# 동시 요청 수 별 성능 비교
# RPS : Request Per Second
from concurrent.futures import ThreadPoolExecutor
from setting import get_llm
import statistics
import time

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘" for i in range(8)
]

def ask(prompt):
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)

    return elapsed, tokens

for workers in [1,2,4,8]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ask, prompts))
    total_time = time.perf_counter() - start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f"\n===== Workers: {workers} =====")

    print("전체 시간 :", round(total_time, 2), "초")
    print("평균 지연 :", round(statistics.mean(latencies), 2))
    print("요청 처리량 :", round(len(results) / total_time, 2), "req/s")
    print("Ouput TPS :", round(total_tokens / total_time, 2))

# Results
# ===== Workers: 1 =====
# 전체 시간 : 6.49 초
# 평균 지연 : 0.81
# Ouput TPS : 123.33

# ===== Workers: 2 =====
# 전체 시간 : 3.65 초
# 평균 지연 : 0.91
# Ouput TPS : 219.21

# ===== Workers: 4 =====
# 전체 시간 : 1.92 초
# 평균 지연 : 0.96
# Ouput TPS : 416.59

# ===== Workers: 8 =====
# 전체 시간 : 1.04 초
# 평균 지연 : 1.03
# Ouput TPS : 772.86