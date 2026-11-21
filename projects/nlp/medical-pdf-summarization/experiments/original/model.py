import re

def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Example usage
input_text = "Sample medical article text with special characters."
cleaned_text = preprocess_text(input_text)

# Example of training BART model
from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Example of fine-tuning BART on medical article summarization dataset
# Code for fine-tuning depends on your dataset and training setup

# Example of using the trained model for summarization
input_text = "Sample medical article text to be summarized."
inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:", summary)

# Example of evaluating summaries using ROUGE
from rouge import Rouge

def evaluate_rouge(reference_summary, generated_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores

# Example usage
reference_summary = "Reference summary of the medical article."
generated_summary = "Generated summary of the medical article."
rouge_scores = evaluate_rouge(reference_summary, generated_summary)
print("ROUGE Scores:", rouge_scores)