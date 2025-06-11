import os 
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import Tool, initialize_agent

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Load API Key
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.0)

# Generic Language Model Tool
prompt = PromptTemplate(
    input_variables=['query'],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

llm_tool = Tool(
    name='language-model',
    func=llm_chain.run,
    description='Use this tool for general queries and logic'
)

# Custom Math Tool (Fallback for `llm-math`)
def simple_math_tool(query):
    try:
        return eval(query)
    except Exception as e:
        return str(e)
    
math_tool = Tool(
    name='Calculator',
    func=simple_math_tool,
    description="Use this tool for basic math calculations."
)

# Tools List
tools = [llm_tool, math_tool]

# Initialize Agent
agent = initialize_agent(
    agent='zero-shot-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

# query = "If I have 54 eggs and Mary has 10, and 5 more people have 12 eggs each. \
#     How many eggs do we have in total?"
query = 'what is the best way to learn Agentic AI tools for profssional job ready skills?'

# Checking the prompt structure
print(agent.agent.llm_chain.prompt.template)
# Run the Agent
result = agent(query)
print(result['output'])