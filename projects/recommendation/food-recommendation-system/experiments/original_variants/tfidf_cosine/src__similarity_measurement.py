import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity_tfidf(df, user_preferred_df, user_preferred_food, top_n=10):
    combined_features = df['Combined_Features'].values
    user_features = user_preferred_df[user_preferred_df['foodItemName'] == user_preferred_food]['Combined_Features'].values
    
    if len(user_features) == 0:
        raise ValueError(f"User preferred food '{user_preferred_food}' not found in the user preferred dataset.")

    all_features = list(combined_features) + list(user_features)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_features)

    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    df['Similarity_Score'] = cosine_sim.flatten()

    df = df.sort_values(by='Similarity_Score', ascending=False).head(top_n)
    
    return df
