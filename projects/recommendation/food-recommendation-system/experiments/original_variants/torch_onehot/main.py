import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.similarity_measurement import calculate_similarity
from src.recommendation_system import recommend_food

def main():
    # Paths to your datasets
    food_data_path = '.\\ai-assignment\\doc\\Food Data.csv'
    user_preferred_data_path = '.\\ai-assignment\\doc\\User Preferances.csv'
    
    # Load and preprocess data
    df, user_preferred_df = load_data(food_data_path, user_preferred_data_path)
    df, user_preferred_df = preprocess_data(df, user_preferred_df)
    
    # Calculate similarity
    df = calculate_similarity(df, user_preferred_df)
    
    # Get recommendations
    recommendations = recommend_food(df, top_n=10)
    
    # Display all columns
    pd.set_option('display.max_columns', None)
    print(recommendations)

if __name__ == '__main__':
    main()
