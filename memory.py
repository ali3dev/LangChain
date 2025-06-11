from config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize Chat Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.6,
                             google_api_key=GOOGLE_API_KEY)

# Test basic chat response
print(llm.invoke("My name is Ali. What is yours?").content)
print(llm.invoke("Great! What is my name?").content)
print(llm.invoke("I am a software engineer. What is your profession?").content)

# Initialize Memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,  # ✅ Fix: "llm=" instead of "chat="
    memory=memory,
    verbose=True
)

# Testing conversation
conversation.invoke("Hello there! I am Ali?")
conversation.invoke("Why is the Sky Blue?")
conversation.invoke("What is the capital of France?")
conversation.invoke("What's my name?")

# Print stored conversation memory
print(memory.buffer)  
print(memory.load_memory_variables({}))