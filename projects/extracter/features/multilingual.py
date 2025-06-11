from langchain.prompts import PromptTemplate
from llm import llm  # Importing LLMClient

def translate_invoice(text, target_language="English"):
    """
    Detects invoice language and translates it into the target language.
    """
    template = """Detect the language of the following invoice and translate it into {target_language}.
    Invoice Text: {invoice_text}

    Provide a clean translated version.
    """
    prompt_template = PromptTemplate(input_variables=["invoice_text", "target_language"], template=template)
    
    response = llm.generate_response(prompt_template.format(invoice_text=text, target_language=target_language))
    
    return response
