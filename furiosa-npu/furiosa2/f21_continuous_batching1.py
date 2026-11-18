import time
import statistics
from concurrent.futures import ThreadPoolExecutor

from kingkeras import run_once

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 자세히 설명해줘"
    for i in range(8)
]

# print(prompts)        # ['NPU 추론 최적화 방법 0번을 자세히 설명해줘', 'NPU 추론 최적화 방법 1번을 자세히 설명해줘', 'NPU 추론 최적화 방법 2번을 자세히 설명해줘', 'NPU 추론 최적화 방법 3번을 자세히 설명해줘', 'NPU 추론 최적화 방법 4번을 자세히 설명해줘', 'NPU 추론 최적화 방법 5번을 자세히 설명해줘', 'NPU 추론 최적화 방법 6번을 자세히 설명해줘', 'NPU 추론 최적화 방법 7번을 자세히 설명해줘']
# print(len(prompts))   # 8

# response = run_once("NPU가 뭐게?", 100)
# print(response["text"])
'''
Okay, the user is asking "NPU가 뭐게?" which is Korean for "What is NPU?" I need to explain what an NPU is. First, I should confirm that NPU stands for Neural Processing Unit, which is a type of specialized processor designed for AI tasks. But wait, sometimes NPU can also refer to something else, like in the context of a company or a specific product. Let me check that.

In the context of technology, NPU
'''

def ask(prompt):
    return run_once(prompt, 100)

for workers in [1, 2, 4, 8]:
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(ask, prompts)
        )

    elapsed = time.perf_counter() - start

    total_tokens = sum(
        r['output_tokens'] for r in results
    )

    # print(len(results))
    # print(total_tokens)

    avg_latency = statistics.mean(
        r["elapsed"] for r in results
    )

    print(f"동시 요청 수: {workers}")
    print(f"전체 시간: {elapsed:.2f}초")
    print(f"전체 출력 토큰: {total_tokens}")
    print(f"처리량: {total_tokens / elapsed:.2f} TPS")
    print(f"평균 요청 지연: {avg_latency:.2f} 초")

'''
동시 요청 수: 1
전체 시간: 8.00초
전체 출력 토큰: 800
처리량: 100.00 TPS
평균 요청 지연: 0.93 초

동시 요청 수: 2
전체 시간: 4.36초
전체 출력 토큰: 800
처리량: 183.61 TPS
평균 요청 지연: 1.09 초

동시 요청 수: 4
전체 시간: 2.22초
전체 출력 토큰: 800
처리량: 360.97 TPS
평균 요청 지연: 1.09 초

동시 요청 수: 8
전체 시간: 1.28초
전체 출력 토큰: 800
처리량: 626.63 TPS
평균 요청 지연: 1.24 초
'''