# Methodology and provenance

## What the original coursework code did

The original `essay_version10.py` loads `EssayText`, `Score1` and `Score2` from a tab-separated training file. It forms `(Score1, Score2)` pairs but selects the first element, `Score1`, as the training label. The model is `BartForSequenceClassification` initialised from `facebook/bart-large-cnn` with four output classes. It uses a maximum input length of 512, batch size 8, AdamW at `5e-5`, a nominal 10-epoch loop, and gradient-scaling/autocast calls for GPU training. It randomly splits rows into train and test partitions with `test_size=0.2` and `random_state=42`.

The historical implementation also removes English stopwords before tokenisation. The cleaned version preserves the original response text apart from the tokenizer's normal handling, because removing function words can alter meaning and grammatical evidence.

## Data audit of the supplied `train.tsv`

The supplied file contained 17,207 rows and columns `Id`, `EssaySet`, `Score1`, `Score2`, and `EssayText`. `EssaySet` values were 1–10. `Score1` and `Score2` values in the inspected file were 0–3. The observed `Score1` counts were:

| Score1 | Rows |
|---:|---:|
| 0 | 6,779 |
| 1 | 5,612 |
| 2 | 4,075 |
| 3 | 741 |

This audit describes the user's uploaded local archive and is not a statement that the dataset may be redistributed.

## Cleaned implementation decisions

- `Score1` remains the target to preserve the source implementation's choice.
- Four labels are declared consistently in the model config.
- Dynamic padding reduces unnecessary padding within each batch.
- A stratified train/validation/test split is used to support checkpoint selection and one-time test evaluation.
- QWK is included because the target values are ordinal.
- The split remains row-level; it does not test generalisation to unseen prompt sets.
- `EssaySet` is recorded in outputs and diagnostics but is not model input in this baseline.

## Reproducibility note

The clean implementation is a reproducible reorganisation and methodological improvement, not a bit-for-bit reproduction of the older notebooks. Do not claim historical metric values for the clean pipeline until the model has been trained and evaluated again using documented splits and configuration.
