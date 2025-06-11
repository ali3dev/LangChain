import os 
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import Tool, initialize_agent

from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer


# Load API Key
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.0)

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name='Search',
        func=docstore.search,
        description='search wikipedia'
    ),
    Tool(
        name='Lookup',
        func=docstore.lookup,
        description='lookup a term in wikipedia'
    )
]

# initialize our agent
docstore_agent  = initialize_agent(
    tools,
    llm,
    agent='react-docstore',
    verbose=True,
    max_iterations=4
)

query = "What were Einstein's main beliefs?"
result = docstore_agent.run(query)
# print(docstore_agent.agent.llm_chain.prompt.template)