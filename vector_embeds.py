import os
from dotenv import find_dotenv, load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chromadb as Chroma

# Load API Key from .env
load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get Gemini Embedding for a single text  
def get_gemini_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

# Define an embedding function for a list of texts
def gemini_embedding_function(texts):
    return [get_gemini_embedding(text) for text in texts]

# Load a PDF file
loader = PyPDFLoader("./data/react-paper.pdf")
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
splits = text_splitter.split_documents(docs)
print(f"Total Chunks: {len(splits)}")

# Create a vector store from documents using our custom Gemini embedding function
persist_directory = "./data/db/chroma"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=gemini_embedding_function,
    persist_directory=persist_directory
)

# Save (persist) the vector store
vectorstore.persist()

# Querying with similarity search
query = "What do they say about ReAct prompting method?"
docs_resp = vectorstore.similarity_search(query=query, k=3)

print(f"Found {len(docs_resp)} relevant chunks!")
print(docs_resp[0].page_content)
