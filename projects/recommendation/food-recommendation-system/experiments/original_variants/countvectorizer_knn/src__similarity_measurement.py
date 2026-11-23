import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

def calculate_similarity_knn(df, user_preferred_df, user_preferred_food, n_neighbors=10):
    vectorizer = CountVectorizer()
    combined_features = df['Combined_Features'].values
    user_features = user_preferred_df[user_preferred_df['foodItemName'] == user_preferred_food]['Combined_Features'].values
    
    if len(user_features) == 0:
        raise ValueError(f"User preferred food '{user_preferred_food}' not found in the user preferred dataset.")

    all_features = list(combined_features) + list(user_features)
    all_features_vectorized = vectorizer.fit_transform(all_features)

    X = all_features_vectorized[:-1]
    user_vector = all_features_vectorized[-1]

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X)
    distances, indices = knn.kneighbors(user_vector, return_distance=True)

    df = df.iloc[indices[0]].copy()
    df['Similarity_Score'] = 1 - distances[0]

    return df
