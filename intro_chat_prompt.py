from llm import llm


customer_review = """I recently purchased a laptop from this store, and it was a terrible experience. The product was defective, the customer service was rude, and they refused to process my refund. I would never recommend this place to anyone. Avoid at all costs!
"""

prompt = f""" Rewrite this {customer_review} in a more professional and neutral tone while keeping the same meaning:
"I recently purchased a laptop from this store, and it was a terrible experience. The product was defective, the customer service was rude, and they refused to process my refund. I would never recommend this place to anyone. Avoid at all costs!"
"""

# without Prompt Template code
rewrite = llm.chat_completion(prompt)
print(rewrite)
print('------------------------------')

# this code for ChatPromptTemplate define in llm.py file to get response (output) not releated to other prompt that define in this...
response = llm.chat_completion("This hotel was dirty and the staff was rude.")
print(response)
print('------------------------------')

cpt = llm.generate_response("This product is terrible and I hate it!")
print(cpt)
