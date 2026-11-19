# Methodology

## Data preparation

Input essays are cleaned by normalizing whitespace and removing empty records. Labels are retained as continuous target scores. The configured text and target columns are read from a CSV file.

## Model

The implementation uses a pretrained BART encoder to obtain contextual representations of essay text. The final encoder state is pooled and passed through a linear regression head to predict a numeric essay score.

## Training

Training uses mean squared error loss, the AdamW optimizer, a warmup scheduler, deterministic seeds, and configurable batch size and sequence length.

## Evaluation

Model quality is measured with mean squared error, root mean squared error, mean absolute error, and Pearson or Spearman correlation when available.
