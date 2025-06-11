from llm import llm
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config import API_PROVIDER, GOOGLE_API_KEY, OPENAI_API_KEY
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Initialize Chat Model
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Sample Email Response
email_response = """
Dear Customer,

Thank you for reaching out. Your order #12345 has been shipped and will arrive in 3-5 business days.
For any queries, please contact support@example.com.

Best regards,  
Customer Support
"""

# Define Output Schema
order_number_schema = ResponseSchema(name="Order Number", description="Order ID, usually numerical. If unavailable, write n/a")
estimated_delivery_schema = ResponseSchema(name="Estimated Delivery Time", description="Expected delivery time, as a list.")
support_email_schema = ResponseSchema(name="Support Email Address", description="Support contact email. If unavailable, write n/a")

response_schema = [order_number_schema, estimated_delivery_schema, support_email_schema]

# Setup the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()  # Get format instructions

# Define Email Template with extracted information
email_template = """
Extract the following information from the email and provide in JSON format with these keys:
- Order Number
- Estimated Delivery Time
- Support Email Address

email: {email}
{format_instructions}
"""

# Format Prompt
prompt_template = ChatPromptTemplate.from_template(email_template)
formatted_prompt = prompt_template.format_messages(email=email_response, format_instructions=format_instructions)

# Generate Response
response = chat(formatted_prompt)

# Print Parsed Output
parsed_response = output_parser.parse(response.content)
print(parsed_response)
