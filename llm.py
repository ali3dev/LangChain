from config import API_PROVIDER, GOOGLE_API_KEY, OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage

class LLMClient:
    def __init__(self):
        if API_PROVIDER == "GOOGLE":
            self.client = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        elif API_PROVIDER == "OPENAI":
            self.client = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        else:
            raise ValueError("Invalid API Provider")

        # ✅ Define ChatPromptTemplate (Structured Messages)
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are an AI assistant."),
            HumanMessagePromptTemplate.from_template("Rewrite this customer review in a professional tone: {user_input}")
        ])

    def generate_response(self, user_input):
        """Generate response using ChatPromptTemplate"""
        formatted_messages = self.chat_prompt.format_messages(user_input=user_input)
        return self.client.invoke(formatted_messages).content

    def chat_completion(self, user_input):
        """Handle OpenAI & Gemini Messages with Role"""
        messages = [
            SystemMessage(content="You are an AI assistant."),  
            HumanMessage(content=user_input)  
        ]
        return self.client.invoke(messages).content  

# ✅ Initialize LLM
llm = LLMClient()
