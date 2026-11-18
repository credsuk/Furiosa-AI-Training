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

for workers in [1, 2, 4, 8]:
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                lambda prompts: run_once(prompts, 50),
                prompts
            )
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
    print(f"평균 요청 지연: {avg_latency:.2f} 초\n")

'''
동시 요청 수: 1
전체 시간: 4.19초
전체 출력 토큰: 400
처리량: 95.56 TPS
평균 요청 지연: 0.45 초

동시 요청 수: 2
전체 시간: 2.33초
전체 출력 토큰: 400
처리량: 171.90 TPS
평균 요청 지연: 0.58 초

동시 요청 수: 4
전체 시간: 1.18초
전체 출력 토큰: 400
처리량: 339.49 TPS
평균 요청 지연: 0.58 초

동시 요청 수: 8
전체 시간: 0.68초
전체 출력 토큰: 400
처리량: 591.94 TPS
평균 요청 지연: 0.65 초
'''