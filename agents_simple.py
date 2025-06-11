import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools

from langchain.agents import Tool, initialize_agent

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# Manually defining a simple math tool instead of `LLMMathChain`
def simple_math_tool(query):
    try:
        return eval(query)
    except Exception as e:
        return str(e)

# llm_math = LLMMathChain.from_llm(llm=llm)
# math_tool = Tool(
#     name="Calculator",
#     func=llm_math.run,
#     description="Useful for when you need to answer questions related to Math."
# )

tools = load_tools(
    ['llm-math'],
    llm=llm
)

print(tools[0].name, tools[0].description)

#ReAct framework = Reasoning and Action
agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3 # to avoid high bills from the LLM
)
query = "If James is currently 45 years old, how old will he be in 50 years? \
    If he has 4 kids and adopted 7 more, how many children does he have?"
result = agent(query)
print(result['output'])

# print(f" ChatGPT ::: {llm.predict('what is 3.1^2.1')}")