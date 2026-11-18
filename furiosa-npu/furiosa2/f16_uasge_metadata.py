
# uasge_metadata > 

from kingkeras import get_llm

llm = get_llm(max_tokens=5000)

response = llm.invoke("NPU란 무엇인가?")

print(response.content)
print("=================================")
print(response.usage_metadata)
# {'input_tokens': 15, 'output_tokens': 1292, 'total_tokens': 1307, 
#  'input_token_details': {'audio': 0, 'cache_read': 6}, 'output_token_details': {'audio': 0, 'reasoning': 0}}





