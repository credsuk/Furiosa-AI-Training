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
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:4 --host 0.0.0.0 --port 8000
"""

from openai import OpenAI
from langchain_openai import ChatOpenAI
import time


def get_llm(*,
        max_tokens=1000,
        temperature=0,
        top_p=0.9,
):
    return ChatOpenAI(
        base_url = "http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

def get_client():
    return OpenAI(
        base_url = "http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )

def get_now_time():
    return time.perf_counter()