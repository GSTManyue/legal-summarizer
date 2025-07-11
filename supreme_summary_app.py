import os
import io
from typing import List, Tuple, Dict

import streamlit as st
import pdfplumber
from transformers import pipeline
import spacy

# -----------------------------
# Caching helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_offline_models():
    """Load offline summarizer (legal‑pegasus) and spaCy model once."""
    try:
        summarizer = pipeline(
            "summarization",
            model="nlpaueb/legal-pegasus",
            tokenizer="nlpaueb/legal-pegasus",
            framework="pt",
            device=0,
        )
    except Exception as e:
        st.error(f"Failed to load legal‑pegasus model: {e}")
        st.stop()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        st.stop()

    return summarizer, nlp


def chunk_text(text: str, max_tokens: int = 1024):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i : i + max_tokens])


def summarize_offline(text: str, summarizer, max_length=256, min_length=80) -> str:
    partial = []
    for chunk in chunk_text(text):
        out = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)[0][
            "summary_text"
        ]
        partial.append(out)
    if len(partial) == 1:
        return partial[0]
    combined = " ".join(partial)
    final = summarizer(combined, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)[0][
        "summary_text"
    ]
    return final


# GPT helper (no cache)

def summarize_with_gpt(text: str, api_key: str, model: str = "gpt-4-turbo") -> str:
    import openai  # imported here to avoid requirement in offline-only mode

    openai.api_key = api_key
    CHUNK_LIMIT = 100_000  # simple safeguard
    if len(text) > CHUNK_LIMIT:
        text = text[:CHUNK_LIMIT]
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a legal document summarizer. Provide a concise summary highlighting key holdings, citations, and relevant legal sections.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# Legal info extraction

def extract_legal_info(text: str, nlp) -> List[str]:
    doc = nlp(text)
    return sorted({ent.text.strip() for ent in doc.ents if ent.label_ in {"LAW", "ORG", "DATE", "GPE"}})


# PDF text extraction

def extract_text_from_pdf(file_bytes: bytes) -> str:
    txt = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            txt += page_text + "\n"
    return txt


# Processing wrapper

def process_pdf(file_bytes: bytes, engine: str, offline_models: Tuple, api_key: str = "") -> Tuple[str, List[str]]:
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        return "[No extractable text]", []

    if engine == "Offline (Legal‑Pegasus)":
        summarizer, nlp = offline_models
        summary = summarize_offline(text, summarizer)
    else:  # OpenAI GPT
        summarizer, nlp = offline_models  # nlp still used for citations
        summary = summarize_with_gpt(text, api_key)

    citations = extract_legal_info(text, nlp)
    return summary, citations


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Supreme Court Order Summarizer", layout="wide", page_icon="⚖️")
st.title("⚖️ Supreme Court Order Summarizer — Hybrid Mode")

st.markdown(
    """Upload one or more **PDF orders** of the Supreme Court of India (English). Choose between **offline summarization** (legal‑pegasus) or **OpenAI GPT** for higher‑quality summaries if you have internet & an API key."""
)

engine = st.selectbox("Choose summarization engine", ["Offline (Legal‑Pegasus)", "OpenAI GPT"])
openai_key = ""
if engine == "OpenAI GPT":
    openai_key = st.text_input("Enter your OpenAI API key", type="password")
    if not openai_key:
        st.warning("API key required for GPT summarization.")

uploaded_files = st.file_uploader("Choose PDF file(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Generate Summaries"):
    with st.spinner("Processing …"):
        offline_models = load_offline_models()
        results: Dict[str, Dict] = {}
        for file in uploaded_files:
            try:
                summary, cites = process_pdf(file.read(), engine, offline_models, openai_key)
                results[file.name] = {"summary": summary, "citations": cites}
            except Exception as e:
                st.error(f"{file.name}: {e}")

    st.header("Results")
    for fname, data in results.items():
        with st.expander(f"📄 {fname}"):
            st.subheader("Summary")
            st.write(data["summary"])
            st.subheader("Citations & Legal Sections")
            st.write(", ".join(data["citations"]) if data["citations"] else "None detected")

    # Download button
    def aggregate_txt(res):
        return "\n\n".join(
            [f"=== {f} ===\nSUMMARY:\n{d['summary']}\n\nCITATIONS:\n{', '.join(d['citations'])}" for f, d in res.items()]
        )

    st.download_button(
        "💾 Download All Summaries (.txt)",
        data=aggregate_txt(results),
        file_name="supreme_court_summaries.txt",
        mime="text/plain",
    )
else:
    st.info("Upload PDFs and click *Generate Summaries* to begin.")
