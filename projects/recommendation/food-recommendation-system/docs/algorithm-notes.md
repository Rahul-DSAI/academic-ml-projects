# Algorithm notes and original variants

The supplied archives contain three related content-similarity experiments over the same 38-row food catalogue and 13-row preference file:

1. **PyTorch cosine baseline** (`torch_onehot`): categorical attributes were label-encoded, concatenated into numeric text, one-hot encoded as whole strings, then compared using cosine similarity. Because the combined encoded string becomes a single sparse token, this does not preserve the semantics of each categorical field reliably.
2. **CountVectorizer + nearest neighbours** (`countvectorizer_knn`): applies a bag-of-words representation and cosine distance to the combined feature strings. Since the strings are numeric label IDs, the default token pattern may discard single-digit tokens; this is a weakness of the original encoding pipeline.
3. **TF-IDF + cosine similarity** (`tfidf_cosine`): the later experiment applies TF-IDF and cosine similarity for one selected preference item. This is the most coherent of the original alternatives, but it inherited the same numeric-label-to-text feature construction and hard-coded Windows paths.

The consolidated implementation keeps the content-based TF-IDF/cosine idea but vectorizes the original descriptive text fields directly: category, item name, item description, tags, dietary label, preparation style, presentation, food concept, flavour and mealtime. It does not use `UserRating` in the similarity vector; ratings are displayed as catalogue metadata and are used only as a tie-break after similarity.

This is a small demonstration catalogue (38 food rows, 13 preference rows), not a production-scale recommendation benchmark. There is no user-interaction history, held-out relevance judgement or feedback dataset in the supplied files. Precision@K, Recall@K, NDCG or claims of recommendation effectiveness should not be reported without such ground truth.
