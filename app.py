import streamlit as st
from summarizer import summarize_document  # your module
import tempfile

st.title("Offline Legal Document Summarizer")

uploaded_pdf = st.file_uploader("Upload a Supreme Court order (PDF)", type="pdf")

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    with st.spinner("Summarizing…"):
        summary, citations = summarize_document(pdf_path)

    st.subheader("Summary")
    st.write(summary)

    st.subheader("Citations & Sections")
    st.write(citations)

