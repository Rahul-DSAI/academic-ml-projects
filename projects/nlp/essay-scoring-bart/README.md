# BART-Based Automated Short-Answer Scoring

Academic NLP project (2024) exploring four-class student-response scoring using `facebook/bart-large-cnn`, PyTorch and Hugging Face Transformers.

## Source implementation

The uploaded coursework implementation:
- reads `EssayText`, `Score1` and `Score2` from `train.tsv`;
- uses `Score1` as the prediction target (the first score only);
- uses BART sequence classification with four classes (`0`, `1`, `2`, `3`);
- uses a maximum sequence length of 512, AdamW, learning rate `5e-5`, batch size 8 and gradient accumulation every 8 batches in the original script;
- trains for 10 epochs and reports accuracy in the original implementation.

## Cleaned implementation in this repository

The `src/` package reorganises the coursework into separate data, model, training, evaluation, metrics and prediction modules. It intentionally differs from the historical script in several ways:

- preserves stopwords instead of removing them, since function words can carry meaning relevant to response scoring;
- uses dynamic batch padding rather than padding every sample to 512 tokens;
- creates stratified train, validation and test splits;
- selects a checkpoint using validation Quadratic Weighted Kappa (QWK), then evaluates once on the held-out test split;
- fixes the label configuration to four classes and records explicit `SCORE_0` through `SCORE_3` mappings;
- saves training history, metrics and predictions locally.

This is a cleaned baseline, not a claim that the historical coursework metrics have been reproduced.

## Dataset

The supplied archive contains the ASAP-SAS `train.tsv` file. Do **not** upload the dataset, leaderboard files, student responses, anchor papers or other competition materials to this public repository unless the applicable competition rules and rights explicitly permit redistribution. Obtain the data directly from the [official Kaggle ASAP-SAS competition](https://www.kaggle.com/competitions/asap-sas/data) and comply with its terms.

Expected local dataset path: `data/raw/train.tsv`.

Required TSV columns: `EssayText`, `Score1`, and `EssaySet`. `Score2` is not used as the target by this implementation. `EssaySet` is retained for split diagnostics and per-prompt analysis, but the current model does not condition on it.

## Installation

Use Python 3.10 or 3.11. Install a PyTorch build appropriate for your hardware from [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining requirements:

```bash
pip install -r requirements.txt
```

## Prepare the dataset

```text
data/
└── raw/
    └── train.tsv
```

The `data/` directory is ignored by Git.

## Train

```bash
python -m src.train --config configs/training_config.yaml
```

Optional overrides:

```bash
python -m src.train --config configs/training_config.yaml --data data/raw/train.tsv --output-dir artifacts
```

The best checkpoint is saved in `artifacts/best_model/`. Training history, split details, test metrics and predictions are also saved under `artifacts/`. These files are local outputs and are ignored by Git.

## Evaluate

```bash
python -m src.evaluate --config configs/training_config.yaml --model-dir artifacts/best_model
```

## Predict

```bash
python -m src.predict --model-dir artifacts/best_model --text "Your sample student response goes here."
```

The model returns a predicted `Score1` class from 0 to 3 and per-class probabilities.

## Metrics

The evaluation script reports:
- accuracy;
- mean absolute error (MAE);
- mean squared error (MSE);
- unweighted Cohen's kappa;
- quadratic weighted kappa (QWK);
- confusion matrix and per-class precision/recall/F1.

QWK is important because the labels are ordered. Metrics from the old write-up should not be copied into this cleaned implementation unless they are reproduced by a documented run.

## Important evaluation limitation

The current split is a row-level random split stratified by `Score1`, broadly following the original coursework split approach. It is **not** a held-out-`EssaySet` evaluation. Since each set may use a different prompt and rubric, report this limitation and consider a separate group-held-out experiment before making claims about generalisation to unseen prompts.

The current model consumes response text only; `EssaySet` is not included as a model feature. This is a limitation for rubric-aware scoring.

## Responsible use

This is an academic experiment, not a validated grading product. Automated predictions should not be used as the sole basis for educational decisions. The model may reflect dataset biases and may not generalise across prompts, student populations or assessment contexts.

## Tests

```bash
pytest -q
```

## Original coursework artifacts

The original ZIP included several notebooks, a Python training script, a model config, project documents, presentation materials, video, research papers and ASAP-SAS data/competition files. Only reviewed source code and materials that can legally be redistributed should be added to this repository. Do not include third-party papers, student response data, competition leaderboard files, client materials or unlicensed model weights.

## Author

Rahul Sinha  
[GitHub](https://github.com/Rahul-DSAI) · [LinkedIn](https://www.linkedin.com/in/rahul-sinha99/)
