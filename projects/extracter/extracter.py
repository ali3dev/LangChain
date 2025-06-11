import json
import re
import pandas as pd
from io import BytesIO
from pypdf import PdfReader
from config import GOOGLE_API_KEY
from llm import llm
from features.multilingual import translate_invoice
from features.categorization import categorize_bill

# ✅ Extract text from PDF
def get_pdf_text(pdf_doc):
    text = ""
    if isinstance(pdf_doc, bytes):
        pdf_reader = PdfReader(BytesIO(pdf_doc))
    else:
        pdf_reader = PdfReader(pdf_doc)
    
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
        else:
            print("⚠️ No text found on a page!")
    
    print("📄 Extracted PDF Text:", text[:500])
    return text

# ✅ Extract structured data from text using LLM
def extracted_data(pages_data):
    template = """Extract the following values from the invoice: 
    Invoice ID, DESCRIPTION, Issue Date, UNIT PRICE, AMOUNT, Bill For, From, Terms.
    
    Return ONLY a valid JSON object with no extra text:
    ```json
    {{
        "Invoice ID": "1001329",
        "DESCRIPTION": "Phone and data bill",
        "Issue Date": "11/27/2026",
        "UNIT PRICE": "500.00",
        "AMOUNT": "500.00",
        "Bill For": "Paul Regex",
        "From": "DR-TeleP",
        "Terms": "Due upon receipt"
    }}
    ```
    """
    full_response = llm.generate_response(template.format(pages=pages_data))
    print("🔍 Raw LLM Response:", full_response)
    
    try:
        # Remove Markdown-style backticks from the response
        clean_json_str = re.sub(r'```json|```', '', full_response).strip()
        llm_extracted_data = json.loads(clean_json_str)
        print("✅ Extracted JSON:", llm_extracted_data)
        return llm_extracted_data
    except json.JSONDecodeError as e:
        print("⚠️ JSON Parsing Error:", e)
        return {}

# ✅ Process multiple PDFs and create DataFrame
def create_docs(user_pdf_list):
    df = pd.DataFrame(columns=["Invoice ID", "DESCRIPTION", "Issue Date", "UNIT PRICE", "AMOUNT", "Bill For", "From", "Terms", "Category"])
    
    for uploaded_file in user_pdf_list:
        pdf_bytes = uploaded_file.read()
        raw_data = get_pdf_text(pdf_bytes)
        
        translated_text = translate_invoice(raw_data, "English")
        llm_extracted_data = extracted_data(translated_text)
        
        print("🔍 Extracted Data from LLM:", llm_extracted_data)
        if isinstance(llm_extracted_data, dict) and llm_extracted_data:
            bill_category = categorize_bill(translated_text)
            llm_extracted_data["Category"] = bill_category
            df = pd.concat([df, pd.DataFrame([llm_extracted_data])], ignore_index=True)
        else:
            print(f"⚠️ Skipping file {uploaded_file.name} due to invalid LLM response.")
    
    print("********************DONE***************")
    return df
