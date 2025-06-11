import os 
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=GOOGLE_API_KEY)

# Load the PDF file
pdf_path = r'D:\Langchain\projects\multidocs\docs\RachelGreenCV.pdf'  # Raw string path to avoid escape issues
pdf_loader = PyPDFLoader(pdf_path)
documents = pdf_loader.load()

# Set up QA chain
chain = load_qa_chain(llm, verbose=True)
query = 'Who is the CV About?'

if not documents:
    print("❌ PDF loading failed!")
else:
    response = chain.run(input_documents=documents, question=query)
    print(response)
