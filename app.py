
import streamlit as st
from summarizer import summarize_text
from citation_extractor import extract_citations
import pdfplumber
import os

# Load logo
st.image("assets/logo.png", width=120)
st.title("Legal Summarizer")
st.caption("by GSTManyue")

# Sidebar Settings
st.sidebar.title("Settings")
use_gpt = st.sidebar.checkbox("Use GPT Hybrid Mode", value=False)
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password") if use_gpt else None
max_summary_length = st.sidebar.slider("Summary Length (words)", 100, 1000, 300)

# File uploader
uploaded_files = st.file_uploader("Upload one or more Supreme Court Orders (PDF)", type="pdf", accept_multiple_files=True)

# Button
if st.button("Summarize"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        for file in uploaded_files:
            st.markdown(f"### 📄 {file.name}")
            with pdfplumber.open(file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text() or ""

            # Summarize
            summary = summarize_text(full_text, use_gpt=use_gpt, api_key=api_key, max_length=max_summary_length)

            # Extract citations
            citations = extract_citations(full_text)

            # Display results
            st.subheader("Summary:")
            st.write(summary)

            if citations:
                st.subheader("Legal Citations/Sections Found:")
                for c in citations:
                    st.markdown(f"- {c}")
            else:
                st.info("No citations or legal sections found.")
