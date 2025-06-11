import pandas as pd
from llm import llm  # Ensure you have the correct LLM module

def query_bills(df, user_question):
    """
    AI will analyze the bills and answer user queries.
    """
    if df.empty:
        return "⚠️ No bill data available to analyze. Please upload a valid bill file."
    
    # Limit data size for performance; sending only first 5 rows
    data_sample = df.head(5).to_dict(orient="records")
    
    # Add extra instruction if query mentions phone bill
    if "phone bill" in user_question.lower():
        extra_instruction = "Only consider entries with the category 'Phone Bill'."
    else:
        extra_instruction = ""
    
    template = """You have the following summarized bill data:
{data}

{extra_instruction}

Answer the user's question in detail:
{question}
"""
    prompt = template.format(data=data_sample, extra_instruction=extra_instruction, question=user_question)
    
    try:
        response = llm.generate_response(prompt)
        return response
    except Exception as e:
        return f"⚠️ Error while processing LLM response: {str(e)}"

if __name__ == "__main__":
    # Example usage: load a sample CSV and query
    import pandas as pd
    file_path = "bills.csv"  # Modify as needed
    df = pd.read_csv(file_path)
    user_question = input("Ask AI about your bills: ").strip()
    if user_question:
        answer = query_bills(df, user_question)
        print("\nAI Response:", answer)
