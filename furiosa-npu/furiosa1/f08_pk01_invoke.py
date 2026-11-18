from furiosa1.a_kingkeras import get_llm

llm = get_llm()
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘.")
print(response.content)
