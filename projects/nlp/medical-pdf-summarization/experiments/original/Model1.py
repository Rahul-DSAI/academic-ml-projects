from transformers import BertTokenizer, BertForMaskedLM
import PyPDF2
import torch

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
model = BertForMaskedLM.from_pretrained("monologg/biobert_v1.1_pubmed")

# Move model to CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def generate_summary(pdf_text):
    # Split input text into smaller chunks if necessary
    chunk_size = 512  # Adjust as needed
    text_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]

    summaries = []
    for chunk in text_chunks:
        input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=512, truncation=True)
        input_ids = input_ids.to(device)  # Move input tensor to the device
        outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    # Concatenate summaries of chunks if needed
    final_summary = " ".join(summaries)
    return final_summary

# Path to the PDF file
pdf_file_path = "C:\\Users\\rahul\\Downloads\\2023.9.JNS231474.pdf" 

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_file_path)

# Generate summary
summary = generate_summary(pdf_text)
print(summary)
