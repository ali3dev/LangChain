import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def generate_insights(df):
    """
    Generates expense insights from extracted data.
    """
    st.subheader("📊 Expense Insights")
    
    # ✅ Total Spending
    if "AMOUNT" in df.columns:
        df["AMOUNT"] = df["AMOUNT"].astype(float)
        total_expense = df["AMOUNT"].sum()
        st.write(f"💰 **Total Expense:** ${total_expense:.2f}")

    # ✅ Expense Distribution
    if "Category" in df.columns:
        st.write("📌 **Category-wise Expense Distribution**")
        fig, ax = plt.subplots()
        df.groupby("Category")["AMOUNT"].sum().plot(kind="bar", ax=ax, color="skyblue")
        st.pyplot(fig)
