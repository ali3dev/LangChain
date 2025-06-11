# load_docs.py
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_community.document_loaders import Docx2txtLoader
except ModuleNotFoundError:
    Docx2txtLoader = None
from langchain_community.document_loaders import TextLoader  # Not used for .txt now
import streamlit as st
import os
from langchain.schema import Document  # For creating Document objects manually

@st.cache_data()
def load_docs():
    documents = []
    docs_dir = r'D:\Langchain\projects\multidocs\docs'
    
    if not os.path.exists(docs_dir):
        st.error("❌ 'docs' folder not found. Please create a 'docs' folder and add your documents.")
        return documents

    for file in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            if Docx2txtLoader is not None:
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            else:
                st.warning(f"Skipping {file}: 'docx2txt' module not installed. Run 'pip install docx2txt'.")
        elif file.endswith('.txt'):
            try:
                with open(file_path, encoding='utf-8', errors='replace') as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": file_path}))
            except Exception as e:
                st.warning(f"Skipping {file}: error loading text file: {e}")
            
    return documents
