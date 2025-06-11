import streamlit as st
import pandas as pd
from extracter import create_docs
from features.analytics import generate_insights
from features.ai_query import query_bills

def main():
    st.set_page_config(page_title="Bill Extractor AI")
    st.title("📄 Bill Extractor AI Assistant 🤖")

    # 1️⃣ Session state for the DataFrame
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()  # Initially empty

    # 2️⃣ Session state for AI answer
    if "ai_answer" not in st.session_state:
        st.session_state.ai_answer = ""

    # Upload Bills
    pdf_files = st.file_uploader("📌 Upload your bills (PDF only)", type=["pdf"], accept_multiple_files=True)
    
    # Button to extract data from PDFs
    extract_button = st.button("🔍 Extract Bill Data")

    if extract_button and pdf_files:
        with st.spinner("⏳ Processing... It may take a moment..."):
            # Create docs & store in session state
            st.session_state.df = create_docs(pdf_files)

        st.success("✅ Processing Done!")

    # Ab hum data_frame ko session_state se use karenge
    data_frame = st.session_state.df

    if not data_frame.empty:
        st.write("🔍 Extracted Data from LLM:", data_frame)

        # Show a preview
        st.write(data_frame.head())

        # Expense Insights
        generate_insights(data_frame)

        # 3️⃣ AI Query System with form
        st.subheader("🤖 Ask AI about your bills")
        with st.form(key="query_form"):
            user_query = st.text_input("🔎 Type your question here")
            submit_button = st.form_submit_button("Get AI Answer")  
            # Enter key will also submit this form

        # If user submitted form + typed a query
        if submit_button and user_query:
            with st.spinner("Processing your query..."):
                st.session_state.ai_answer = query_bills(data_frame, user_query)

        # Display AI answer (if we have one)
        if st.session_state.ai_answer:
            st.write("💬 **AI Response:**", st.session_state.ai_answer)

        # Download CSV
        convert_to_csv = data_frame.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", convert_to_csv, "Bills.csv", "text/csv", key="download-csv")

    else:
        st.warning("⚠️ No valid data extracted yet. Please upload and extract your PDF bills.")

if __name__ == '__main__':
    main()
