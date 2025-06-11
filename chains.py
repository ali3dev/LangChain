from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from config import GOOGLE_API_KEY
from llm import llm

## ==== Using Google GenAI API (Gemini) =======
llm_model = 'gemini-1.5-flash'

# Chat Model (Conversational AI)
chat = ChatGoogleGenerativeAI(model=llm_model,
                            temperature=0.6,
                            google_api_key=GOOGLE_API_KEY)

# General LLM Model (Non-Chat)
google_llm = GoogleGenerativeAI(model=llm_model, temperature=0.7, google_api_key=GOOGLE_API_KEY)

# LLMChain
prompt = PromptTemplate(
    input_variables=['language'],
    template='How do you say good morning in {language}?'
)

chain = LLMChain(llm=google_llm, prompt=prompt, verbose=True)

# Running the chain with German input
print(chain.run(language="German"))