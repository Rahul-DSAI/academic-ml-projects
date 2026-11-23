def recommend_food(df, top_n=10):
    recommended_foods = df.sort_values(by='Similarity_Score', ascending=False).head(top_n)
    return recommended_foods[['foodItemName', 'restaurantName', 'UserRating']]
