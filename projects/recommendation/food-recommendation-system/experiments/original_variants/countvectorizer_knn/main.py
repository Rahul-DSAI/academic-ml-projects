from src.data_preprocessing import load_data, preprocess_data
from src.similarity_measurement import calculate_similarity_knn
from src.recommendation_system import recommend_food

def main():
    # Paths to your datasets
    food_data_path = 'D:\\Streamlit\\Food_Recommendation_System1\\ai-assignment\\doc\\Food Data.csv'
    user_preferred_data_path = 'D:\\Streamlit\\Food_Recommendation_System1\\ai-assignment\\doc\\User Preferances.csv'
    
    # Load and preprocess data
    df, user_preferred_df = load_data(food_data_path, user_preferred_data_path)
    df, user_preferred_df = preprocess_data(df, user_preferred_df)
    
    # Get user preferred food item
    user_preferred_food = 'Lunch Sashimi'
    
    # Calculate similarity using KNN
    try:
        similar_foods_df = calculate_similarity_knn(df, user_preferred_df, user_preferred_food, n_neighbors=10)
    except ValueError as e:
        print(e)
        return
    
    # Get recommendations
    recommendations = recommend_food(similar_foods_df, top_n=10)
    print(recommendations)

if __name__ == '__main__':
    main()