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
"""

from openai import OpenAI


client = OpenAI(
    base_url = "http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

# 서버에 로드도니 모델 확인
models = client.models.list()
model_id = models.data[0].id

print("모델 : ", model_id)


#  추론
response = client.chat.completions.create(
    model = model_id,
    messages = [
        {
            "role" : "user",
            "content" : "한국에 국가를 알어?",

        }
    ],
    max_tokens=3000,
)

print(response.choices[0].message.content)



