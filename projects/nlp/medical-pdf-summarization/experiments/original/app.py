import streamlit as st
import re
from transformers import BartTokenizer, BartForConditionalGeneration
from PyPDF2 import PdfReader

def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def summarize_text(input_text):
    # Initialize tokenizer and model with BioBART
    tokenizer = BartTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')
    model = BartForConditionalGeneration.from_pretrained('monologg/biobert_v1.1_pubmed')
    
    # Preprocess input text
    cleaned_text = preprocess_text(input_text)

    # Generate summary
    input_ids = tokenizer(cleaned_text, return_tensors='pt', max_length=1024, truncation=True).input_ids
    summary_ids = model.generate(input_ids, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    st.title("Medical Text Summarization")

    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    if uploaded_file is not None:
        # Read PDF file
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.subheader("Original Text:")
        st.text_area("Original Text", text, height=300)

        summary = summarize_text(text)

        st.subheader("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
