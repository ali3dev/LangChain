import os 
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.text_splitter import CharacterTextSplitter

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm_model = "gemini-1.5-flash"
chat  = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)

## 1. CharacterTextSplitter
with open("./data/i-have-a-dream.txt", encoding="utf-8") as paper:
    speech = paper.read()
    
text_splitter = CharacterTextSplitter(
    
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len
)

texts = text_splitter.create_documents([speech])
print(texts[0])