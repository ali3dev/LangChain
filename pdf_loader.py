import os 
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from langchain.document_loaders import PyPDFLoader

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm_model = "gemini-1.5-flash"
chat  = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)

### pip install pypdf

loader = PyPDFLoader('./data/react-paper.pdf')
pages = loader.load()

# print(len(pages))
page = pages[0]
print(page.page_content[0:700])
print(page.metadata)