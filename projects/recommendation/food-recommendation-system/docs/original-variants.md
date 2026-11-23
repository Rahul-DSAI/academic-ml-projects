# Original archive comparison

- `Food_Recommendation_System.zip` and `Food_Recommendation_System(1).zip` contain the same 40-entry archive content.
- `Food_Recommendation_System1.zip` contains a large bundled Windows Python/Conda environment under `ai-assignment/foodRS/` (over 6,000 entries), plus source variants for PyTorch cosine, CountVectorizer/kNN and TF-IDF/cosine. The bundled environment is not source code and is intentionally not copied into this repository.
- `Food_Recommendation_System2.zip` contains the TF-IDF/cosine iteration with absolute Windows paths.

The `experiments/original_variants/` folder keeps the original `main.py` and source modules from the three algorithm iterations, stripped of `.git`, `__pycache__`, and the embedded environment.
