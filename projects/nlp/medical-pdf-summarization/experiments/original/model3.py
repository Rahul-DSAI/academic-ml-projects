import re
import nltk
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

# Function to preprocess text
def preprocess_text(text):
    # Remove references (e.g., "[1]") and figure/table captions
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"Figure \d+:", "", text)
    text = re.sub(r"Table \d+:", "", text)

    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_sentences.append(" ".join(filtered_words))
    
    return filtered_sentences

# Function to load BioBERT tokenizer and model
def load_biobert():
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
    model = model.to(device)  # Move model to GPU if available
    return tokenizer, model

# Function to calculate sentence embeddings
def calculate_embeddings(tokenizer, model, sentences):
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    tokenized_sentences = tokenized_sentences.to(device)  # Move tensors to GPU if available
    with torch.no_grad():
        outputs = model(**tokenized_sentences)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Extract embeddings for [CLS] token
    return embeddings

# Function to perform extractive summarization
def extractive_summarization(sentences, embeddings):
    similarity_matrix = cosine_similarity(embeddings.cpu().numpy())  # Move embeddings back to CPU for cosine_similarity
    sentence_scores = np.mean(similarity_matrix, axis=1)
    top_sentences_indices = np.argsort(sentence_scores)[-3:][::-1]  # Get indices of top-scoring sentences
    top_sentences = [sentences[index] for index in top_sentences_indices]  # Get top-scoring sentences
    return top_sentences

# PDF file path
pdf_path = "C:\\Users\\rahul\\Downloads\\2023.9.JNS231474.pdf"

# Extract text from PDF
raw_text = extract_text_from_pdf(pdf_path)

# Preprocess text into sentences
sentences = preprocess_text(raw_text)

# Load BioBERT tokenizer and model
tokenizer, model = load_biobert()

# Calculate sentence embeddings
embeddings = calculate_embeddings(tokenizer, model, sentences)

# Perform extractive summarization
summary = extractive_summarization(sentences, embeddings)

# Print summary
print("Extractive Summary:")
for sentence in summary:
    print(sentence)
