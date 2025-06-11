# app.py
import streamlit as st
from helper import search_serp, pick_best_articles_urls, extract_content_from_urls, summarizer, generate_newsletter

def main():
    st.set_page_config(page_title="AI-Powered Newsletter", layout="centered")
    st.title("AI-Powered Newsletter Generator")
    st.write("Enter a topic and get a newsletter summarizing the latest news.")

    topic = st.text_input("Enter Topic:", placeholder="e.g., Tech News, AI advancements...")
    process_btn = st.button("Generate Newsletter")

    if process_btn and topic:
        st.info("Processing your request... Please wait.")
        try:
            # ---- Step 1: Search Results ----
            st.subheader("Step 1: Search Results")
            search_results = search_serp(topic)
            st.write("Raw Search Results:", search_results)

            # ---- Step 2: Selected Article URLs ----
            st.subheader("Step 2: Selected Article URLs")
            article_urls = pick_best_articles_urls(search_results, topic)
            st.write("Selected URLs:", article_urls)
            if not article_urls:
                st.warning("No relevant articles found. Try a different topic.")
                return

            # ---- Step 3: Extracted Content ----
            st.subheader("Step 3: Extracted Content")
            extracted_contents = extract_content_from_urls(article_urls)
            # If using FAISS, we can try to display number of vectors
            try:
                total_vectors = extracted_contents.index.ntotal
            except Exception:
                total_vectors = "N/A"
            st.write(f"Number of document chunks (vectors): {total_vectors}")

            # ---- Step 4: Summarized Articles ----
            st.subheader("Step 4: Summarized Articles")
            summarized_articles = summarizer(extracted_contents, topic)
            st.write("Summaries:", summarized_articles)

            # ---- Step 5: Generated Newsletter ----
            st.subheader("Step 5: Generated Newsletter")
            newsletter_text = generate_newsletter(summarized_articles, topic)
            st.success("Newsletter generated successfully!")
            st.write(newsletter_text)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    elif process_btn:
        st.warning("Please enter a topic.")

if __name__ == "__main__":
    main()
