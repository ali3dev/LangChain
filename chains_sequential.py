import os 
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm_model = "gemini-1.5-flash"
open_ai = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)

template = """ 
 As a children's book writer, please come up with a simple and short (90 words)
 lullaby based on the location
 {location}
 and the main character {name}
 
 STORY:
"""

prompt = PromptTemplate(input_variables=["location", "name"], template=template)
chain_story = LLMChain(llm=open_ai, 
                       prompt=prompt,
                       output_key='story',
                       verbose=True)

story = chain_story({"location": "in the forest", "name": "Bobby"})
print(story)

# ======= Sequential Chain =====
# chain to translate the story to Portuguese

template_update = """
Translate the {story} into {language}.  Make sure 
the language is simple and fun.

TRANSLATION:
"""


prompt_translate = PromptTemplate(input_variables=['story', 'language'],
                                  template=template_update)

chain_translate = LLMChain(
    llm=open_ai,
    prompt=prompt_translate,
    output_key='translated'
)

# ==== Create the Sequential Chain ===

overall_chain = SequentialChain(
    chains=[chain_story, chain_translate],  
    input_variables=['location', 'name', 'language'],
    output_variables=['story', 'translated'],  
    verbose=True
)


response = overall_chain({
    'location': 'in the forest',
    'name': 'Bobby',
    'language': 'Portuguese'    
})


print(f"English Version ====> { response['story']} \n \n")
print(f"Translated Version ====> { response['translated']}")