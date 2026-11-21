"""Streamlit interface for medical PDF extractive summarization."""

from __future__ import annotations

import streamlit as st
from pypdf import PdfReader
from io import BytesIO

from medical_summarizer.embeddings import BioBERTEmbedder
from medical_summarizer.summarizer import summarize_text


st.set_page_config(page_title="Medical PDF Summarizer", page_icon="📄", layout="wide")
st.title("Medical PDF Summarizer")
st.caption("Extractive summarization with BioBERT sentence embeddings")

st.info(
    "This prototype selects sentences from the source PDF. It does not generate "
    "new medical advice and is not a clinical decision-support system."
)


@st.cache_resource(show_spinner="Loading BioBERT model…")
def load_embedder() -> BioBERTEmbedder:
    return BioBERTEmbedder()


uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
num_sentences = st.slider("Summary length (sentences)", min_value=1, max_value=15, value=5)

if uploaded is not None:
    try:
        reader = PdfReader(BytesIO(uploaded.getvalue()))
        raw_text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if not raw_text:
            st.error("No selectable text found. This may be a scanned PDF; OCR is not included.")
        else:
            with st.expander("Extracted source text"):
                st.text_area("PDF text", raw_text, height=260)
            if st.button("Generate summary", type="primary"):
                with st.spinner("Encoding sentences and ranking salient content…"):
                    summary = summarize_text(
                        raw_text,
                        load_embedder(),
                        num_sentences=num_sentences,
                    )
                st.subheader("Extractive summary")
                st.write(summary)
                st.download_button(
                    "Download summary",
                    data=summary,
                    file_name="medical_summary.txt",
                    mime="text/plain",
                )
    except Exception as exc:
        st.error(f"Unable to process this PDF: {exc}")
