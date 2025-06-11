from langchain.prompts import PromptTemplate
from llm import llm  # Importing LLMClient

def categorize_bill(text):
    """
    Categorizes the bill based on its content.
    """
    template = """Read the following invoice and determine its category from the list:
['Electricity Bill', 'Water Bill', 'Phone Bill', 'Internet Bill', 'Shopping Receipt', 'Others'].
If the invoice text contains keywords related to telephone or mobile usage (like 'phone', 'mobile', 'call', etc.), return 'Phone Bill'.

Invoice Text: {invoice_text}

Return only the category name.
"""
    prompt_template = PromptTemplate(input_variables=["invoice_text"], template=template)
    category = llm.generate_response(prompt_template.format(invoice_text=text))
    return category.strip()

if __name__ == "__main__":
    sample_text = "This invoice is from XYZ Telecom for phone calls and data usage."
    print(categorize_bill(sample_text))
