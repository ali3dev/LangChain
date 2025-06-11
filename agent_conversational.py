import os 
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import Tool, initialize_agent

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


# Load API Key
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.0)

# Memory 
memory = ConversationBufferMemory(memory_key='chat_history')

prompt = PromptTemplate(
    input_variables=['query'],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Initialize the LLM Tool
llm_tool = Tool(
    name='language-model',
    func=llm_chain.run, 
    description='Use this tool for general queries and logic'
)

tools = load_tools(
    ['llm-math'],
    llm=llm
)

tools.append(llm_tool)

# Conversational Agent
conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=llm,
    max_iterations=3,
    memory=memory,
    verbose=True,
)

query = "How old is a person born in 1917 in 2023"
    
query_two = "How old would that person be if their age is multiplied by 100?"
    
print(conversational_agent.agent.llm_chain.prompt.template)

result = conversational_agent(query)
results = conversational_agent(query_two)
# print(result['output'])