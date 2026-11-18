"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 > 가능
furiosa-llm serve furiosa-ai/gpt-oss-20b  > 가능
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic > 가능
furiosa-llm serve Qwen/Qwen3.5-4B > 가능
furiosa-llm serve Qwen/Qwen3-4B > 가능 
furiosa-llm serve Qwen/Qwen3-8B-FP8 --fx ./qwen3-8b-fp8.fxb > 컴파일한 파일

furiosa-llm serve furiosa-ai/Qwen3-VL-4B-Thinking > 가능 / 이미지, 영상 분석
furiosa-llm serve furiosa-ai/e5-mistral-7b-instruct > 가능 / 임베딩

furiosa-llm serve furiosa-ai/EXAONE-4.0-32B-FP8 > 불가능
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct > 불가능

LLM 연결
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:4 --host 0.0.0.0 --port 8000 --enable-overlap-scheduling --enable-auto-tool-choice --tool-call-parser hermes

동시 요청할때는 오버랩 / 단일 요청시에는 불필요
--enable-overlap-scheduling

Tools를 사용하기 위한 옵션
--enable-auto-tool-choice 
--tool-call-parser hermes
"""



from langchain_openai import ChatOpenAI

def get_llm(max_tokens=None, temperature=None, top_p=None):
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        max_tokens = max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return llm


import time
def run_once(prompt, max_tokens=1000):
    llm = get_llm(max_tokens)

    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {}

    return {
        "elapsed" : elapsed,
        # "input_tokens" : usage['input_tokens'],  dict에 null이 들어가면 에러 발생
        "input_tokens" : usage.get('input_tokens', 0), # null이면 0을 반환하기 위해서 get 함수 사용
        "output_tokens" : usage.get('output_tokens', 0),
        "text" : response.content,
    }


import urllib.request
def get_cache_metrics():
    url = "http://127.0.0.1:8000/metrics"

    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read().decode("utf-8")

    hits = 0
    queries = 0
    kv_usage = 0

    for line in data.splitlines():
        if line.startswith('#'):
            continue
        if "furiosa_llm_prefix_cache_hits{" in line:
            hits += float(line.split()[-1])
        elif "furiosa_llm_prefix_cache_queries{" in line:
            queries += float(line.split()[-1])
        elif "furiosa_llm_kv_cache_usage_percent{" in line:
            kv_usage = float(line.split()[-1]) * 100

    hit_rate = hits / queries * 100 if queries > 0 else 0

    print("========Furiosa NPU Cache 상태 ========")
    print(f"Prefix Cache 조회 횟수: {queries:.0f}")
    print(f"Prefix Cache 적중 횟수: {hits:.0f}")
    print(f"Prefix Cache 적중률: {hit_rate:.2f}")
    print(f"KV Cache 사용률: {kv_usage:.2f}")

    return{
        "queries" : queries,
        "hits": hits,
        "hit_rate": hit_rate,
        "kv_usage": kv_usage,
    }

# get_cache_metrics()
'''
========Furiosa NPU Cache 상태 ========
Prefix Cache 조회 횟수: 7287
Prefix Cache 적중 횟수: 0
Prefix Cache 적중률: 0
KV Cache 사용률: 1
'''