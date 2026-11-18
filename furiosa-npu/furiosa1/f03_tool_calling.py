"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:4 --host 0.0.0.0 --port 8000
"""
import json
import os
from openai import OpenAI

base_rul = os.getenv("OPENAI_BASE_URL", 'http://localhost:8000/v1')
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
client = OpenAI(base_url=base_rul, api_key=api_key)


def get_weather(location:str, unit:str):
    return f"Getting the weater for {location} in {unit}"

def get_time(timezone:str):
    return f"Gettring the current time in {timezone}..."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone, e.g., 'America/Los_Angeles'"}
                },
                "required": ["timezone"]
            }
        }
    }
]



#  추론
response = client.chat.completions.create(
    model = client.models.list().data[0].id,
    messages = [
        {
            "role" : "user",
            "content" : "How to weather LA?",
        }
    ],
    tools=tools,
    tool_choice={"type":"function", "function": {"name":"get_weather"}},
    max_tokens=3000,
)

tool_calls = response.choices[0].message.tool_calls[0].function
print(f"function called: {tool_calls.name}")
print(f"Arguments : {tool_calls.arguments}")
assert tool_calls.name == "get_weather", f"Expected 'get_weather' but got '{tool_calls.name}"
print(f"Result: {get_weather}",  f"Expected 'get_weather' but got '{tool_calls.name}'")

print(response.choices[0].message.content)


