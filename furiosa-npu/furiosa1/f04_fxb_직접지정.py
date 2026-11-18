# pip install furiosa_llm
from furiosa_llm import LLM, SamplingParams


with LLM("Qwen/Qwen3-8B-FP8", fxb="/root/qwen3-8b-fp8.fxb") as llm:
    params = SamplingParams(max_tokens=10000, top_p=0.3, top_k=100)

    message = [{
            "role" : "user",
            "content" : "오늘 엔비디아 주가각 어떻게 되?",
        }]
    prompt = llm.tokenizer.apply_chat_template(message, tokenize=False)

    response = llm.generate([prompt], params)
    print(response[0].outputs[0].text)



