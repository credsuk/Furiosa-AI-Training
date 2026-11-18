from kingkeras import run_once, get_cache_metrics

prefix = "NPU 추론 최적화에 대한 설명입니다." * 150 # "문자열"을 150번 곱해서 써라
# 2번 돌렸을때 프리픽스 캐시가 없다면 동일하게 나간다


for i in range(1, 5):
    prompt = prefix + f"\n질문 {i}: KV Cache란?"

    # 요청 전 캐시 상태
    before = get_cache_metrics()
    result = run_once(prompt)
    after = get_cache_metrics()

    # 이번 요청에서 증가한 횟수
    hits = after["hits"] - before["hits"]
    queries = after["queries"] - before["queries"]

    if queries > 0 :
        hit_rate = hits / queries * 100
    else:
        hit_rate = 0

    print(f"\n===== {i}회 요청 =====")
    print("응답 시간 :", round(result["elapsed"], 3))
    print("입력 토큰 :", result["input_tokens"])
    print("이번 조회 횟수:", int(queries)) # 이건 안전빵으로 형변환 넣어준거
    print("이번 적중 횟수:", int(hits))
    print("이번 적중률: ", round(hit_rate, 2), "%")
