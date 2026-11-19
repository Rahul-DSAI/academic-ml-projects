# Essay Scoring with BART

A reproducible NLP project for automated essay scoring using a sequence-to-sequence BART encoder and a regression-style prediction head.

## Project structure

```text
essay-scoring-bart/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/training_config.yaml
├── src/
├── notebooks/historical_experiments.ipynb
├── tests/
├── assets/evaluation_plots/
├── docs/methodology.md
└── results/README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Place your essay CSV as configured in `configs/training_config.yaml`, then run:

```bash
python src/train.py --config configs/training_config.yaml
```

## Prediction

```bash
python src/predict.py --model_dir results/best_model --essay "Write your essay here"
```

## Testing

```bash
pytest tests
```
