from kingkeras import run_once, get_cache_metrics

prompts =[
    "NPU란 무엇인가?",
    "NPU의 특징을 설명해줘" * 100,
    "NPU의 특징을 설명해줘" * 300,
]

for i, prompt in enumerate(prompts, 1):
    # print(i)
    print(f"\n============ Test {i} ============")

    result = run_once(prompt, 300)
    print("입력 토큰: ", result["input_tokens"])
    print("출력 토큰: ", result["output_tokens"])
    print("응답 시간: ", round(result["elapsed"], 2))

    print("\nKV Cache 메트릭")
    get_cache_metrics()


'''
============ Test 1 ============
입력 토큰:  15
출력 토큰:  300
응답 시간:  2.82

KV Cache 메트릭
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 10929
Prefix Cache 적중 횟수: 6394
Prefix Cache 적중률: 58.50
KV Cache 사용률: 2.20

============ Test 2 ============
입력 토큰:  908
출력 토큰:  300
응답 시간:  2.89

KV Cache 메트릭
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 11836
Prefix Cache 적중 횟수: 7301
Prefix Cache 적중률: 61.68
KV Cache 사용률: 2.29

============ Test 3 ============
입력 토큰:  2708
출력 토큰:  300
응답 시간:  2.85

KV Cache 메트릭
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 14543
Prefix Cache 적중 횟수: 10008
Prefix Cache 적중률: 68.82
KV Cache 사용률: 2.37
'''