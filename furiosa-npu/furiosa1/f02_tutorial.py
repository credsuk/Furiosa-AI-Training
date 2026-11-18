# pip install furiosa_llm
from furiosa_llm import LLM, SamplingParams


# llm = LLM("furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic")  >  이뜻
with LLM("furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic") as llm:
    params = SamplingParams(max_tokens=10000, top_p=0.3, top_k=100)

    message = [{
            "role" : "user",
            "content" : "오늘 엔비디아 주가각 어떻게 되?",
        }]
    prompt = llm.tokenizer.apply_chat_template(message, tokenize=False)

    response = llm.generate([prompt], params)
    print(response[0].outputs[0].text)



