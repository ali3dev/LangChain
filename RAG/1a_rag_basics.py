import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Set file and vector store path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")
os.makedirs(persistent_directory, exist_ok=True)

# Step 2: Check if vector DB exists
if not os.listdir(persistent_directory):
    print("Persistent directory is empty. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Step 4: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Step 5: Create embeddings using HuggingFace
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    print("\n--- Finished creating embeddings ---")

    # Step 6: Save to vector DB
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. Loading and inspecting...")

    # Load the vector store
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Get number of vectors/documents
    print("Number of vectors:", db._collection.count())

    # Fetch and print a sample document
    docs = db.similarity_search("Odyssey", k=1)
    print("Sample document:", docs[0].page_content)