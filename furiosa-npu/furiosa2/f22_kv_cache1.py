from kingkeras import run_once

short_prompt = "NPU란 무엇인가"
long_prompt = (
    "NPU는 인공지능 연산을 가속하는 프로세서다." * 100
) + "'위 내용을 요약해줘"

tests = [
    ("짧은 입력", short_prompt, 80),
    ("긴 입력", long_prompt, 80),
    ("긴 입력과 긴 출력", long_prompt, 400)
]

for name, prompt, max_tokens in tests:
    result = run_once(prompt, max_tokens)

    print(f"\n실험: {name}")
    print(f"입력 토큰: ", result["input_tokens"])
    print(f"출력 토큰: ", result["output_tokens"])
    print(f"소요 시간: ", round(result["elapsed"], 2))
    input("서버 KV Cache 사용률 확인해보기 [Enter]")    # 파워셸에서  RNGD KV cache usage: 직접 확인
