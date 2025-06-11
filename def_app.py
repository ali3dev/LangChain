from llm import llm

# Example Prompt
prompt = "What is the weather in Pakistan Bannu?"

# Get Response
response = llm.generate_response(prompt)
print(response.content)
