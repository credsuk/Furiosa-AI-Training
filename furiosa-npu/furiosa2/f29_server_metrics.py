import requests

url = "http://127.0.0.1:8000/metrics"

response = requests.get(url, timeout=5)
response.raise_for_status()

# print(response) # <Response [200]>

keywords = [
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_percent",
    "e2e_request_latency_seconds"
]

for line in response.text.splitlines():
    if line.startswith("#"):
        continue
    if any(word in line for word in keywords):
        print(line)

# Results
# furiosa_llm_num_requests_running{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0"} 0
# furiosa_llm_kv_cache_usage_percent{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0"} 0.14156206652128348
# furiosa_llm_num_requests_waiting{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0"} 0
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="0.3"} 0
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="0.5"} 0
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="0.8"} 3
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="1"} 36
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="1.5"} 44
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="2"} 47
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="2.5"} 47
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="5"} 48
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="10"} 85
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="15"} 85
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="20"} 85
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="30"} 85
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="40"} 85
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="50"} 85
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="60"} 85
# furiosa_llm_e2e_request_latency_seconds_bucket{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0",le="+Inf"} 85
# furiosa_llm_e2e_request_latency_seconds_sum{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0"} 362.40680233
# furiosa_llm_e2e_request_latency_seconds_count{model_name="furiosa-ai/Qwen3-4B-FP8",engine="0"} 85