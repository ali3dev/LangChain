# doc_vector.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Using FAISS instead of Chroma
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=GOOGLE_API_KEY)

# Specify the PDF file path (using the docs folder)
pdf_path = "D:/Langchain/projects/multidocs/docs/RachelGreenCV.pdf"

# Check if the file exists
if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"File not found at {pdf_path}. Please check the file path and ensure the file exists.")

# Load the PDF file
pdf_loader = PyPDFLoader(pdf_path)
documents = pdf_loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)
if not docs:
    raise ValueError("No document chunks generated. Check your documents or splitting parameters.")

# Initialize embeddings with API key & model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Correct model name
    google_api_key=GOOGLE_API_KEY   # API key
)

# Create our vector database using FAISS (no persistence by default)
vectordb = FAISS.from_documents(
    documents=docs,
    embedding=embedding
)

# Use RetrievalQA chain to get info from the vectorstore
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True
)

# Supply the correct input key "query"
result = qa_chain.invoke({"query": "when did Rachel graduate?"})
print(result)

print("Total vectors in index:", vectordb.index.ntotal)

first_embedding = vectordb.index.reconstruct(0)
print("Embedding for first document chunk:", first_embedding)
