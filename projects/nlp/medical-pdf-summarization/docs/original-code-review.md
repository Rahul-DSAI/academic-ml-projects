# Review of the original TEXTSAMMARIZATION archive

The ZIP contains six Python files. They represent several partially implemented approaches, not one consistent end-to-end system.

## Files and findings

| Original file | Intended role | Important observation |
|---|---|---|
| `app.py` | Streamlit medical PDF summarizer | Uses `BartForConditionalGeneration` with `monologg/biobert_v1.1_pubmed`. BioBERT is a BERT encoder checkpoint, not a BART encoder-decoder checkpoint; these classes/weights are incompatible. |
| `Datacollection.py` | Example article data download | Uses `https://api.example.com/medical_articles`, a placeholder endpoint; it is not a working PubMed integration. |
| `model.py` | BART experiment and ROUGE example | Loads `facebook/bart-large-cnn`, but contains no fine-tuning pipeline or project-specific trained weights. |
| `Model1.py` | PDF chunk summarization | Loads `BertForMaskedLM` and calls `.generate()` as if it were a summarization model. Masked-language modeling is not the same as sequence-to-sequence summarization. |
| `Model2.py` | BioBERT extractive summarization | Uses BioBERT sentence embeddings, cosine similarity, and selects the three highest-centrality sentences. |
| `model3.py` | Refined extractive experiment | Adds GPU support and removes citations/figure/table labels and English stopwords before embedding. Removing stopwords can damage sentence meaning and is not retained in the cleaned pipeline. |

## Maintained implementation

The `src/medical_summarizer/` package formalizes the extractive approach used in `Model2.py` and `model3.py`:

1. Extract selectable PDF text with `pypdf`.
2. Split the text into sentences.
3. Generate BioBERT contextual embeddings.
4. Mean-pool token embeddings using the attention mask.
5. Compute pairwise cosine similarity.
6. Rank sentences by mean similarity to the other sentences.
7. Select the requested number of sentences and restore source order.

No model fine-tuning is performed. No generative BART summarizer is claimed.

## Reproducibility and evaluation gaps

The archive contains a hard-coded local PDF path, but no PDF, dataset, reference summaries, trained project checkpoint, or recorded benchmark metrics. The included ROUGE code is only an example and cannot establish performance without reference summaries.

The maintained implementation therefore makes no accuracy or ROUGE claim. A meaningful evaluation should use a document/reference-summary dataset, compute ROUGE-1/2/L, compare with a lead baseline, and include human assessment of factual consistency and coverage.
