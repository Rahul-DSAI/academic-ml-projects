# Review of the supplied notebooks

## Uploaded files

- `sentimentanalysis.ipynb`: contains a one-cell PyTorch/Hugging Face BERT classifier.
- `sentiment.ipynb`: contains a Kaggle download cell and almost identical BERT code, with the data root changed to `/content/unprocessed/sorted_data`.
- `Sentiment_Analysis.ipynb`: contains one empty code cell and no executable implementation.

The two populated training implementations are approximately 99.4% text-similar. Both use the 25 product-domain folder names as integer labels and train `BertForSequenceClassification` with 25 output classes.

## Task-name correction

Although the notebook filenames say *Sentiment Analysis*, the code assigns labels with `categories.index(category)` while iterating through product-domain folders. Therefore, its supervised target is the product domain (for example, books, apparel, electronics), not positive/negative/neutral sentiment. The code does not extract a sentiment-polarity target from the review records. The cleaned project is consequently named **BERT Multi-Domain Product Review Classification**.

A true sentiment-polarity classifier would need explicit polarity labels (or a documented, reliable parsing procedure for the source dataset's labelled records) and a separate label mapping. This project does not claim to implement that task.

## Improvements in the cleaned pipeline

- Uses one configurable dataset root rather than a hard-coded Kaggle/Colab path.
- Uses a deterministic, stratified train/validation/test split.
- Reserves the test partition for final evaluation instead of reporting test accuracy after every epoch.
- Uses dynamic batch padding and a configurable maximum sequence length.
- Does not remove stopwords before BERT tokenization; negations and other function words can be important.
- Uses `torch.optim.AdamW` and a linear learning-rate schedule.
- Selects the checkpoint by validation macro-F1 and writes a held-out test report.
- Saves model, tokenizer, label mapping, split summary and metrics.

## Limitations

- The original dataset is not included. Download it separately and review the dataset provider's current license and terms.
- The notebook does not contain a trustworthy completed classification report or verifiable model checkpoint.
- No performance metric is claimed until the cleaned pipeline is trained and evaluated.
- This is domain classification of reviews, not sentiment polarity prediction.
