import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(food_data_path, user_preferred_data_path):
    df = pd.read_csv(food_data_path)
    user_preferred_df = pd.read_csv(user_preferred_data_path)
    return df, user_preferred_df

def preprocess_data(df, user_preferred_df):
    df.fillna('Unknown', inplace=True)
    user_preferred_df.fillna('Unknown', inplace=True)

    categorical_columns = ['tags', 'Dietary', 'FoodPreparationStyles', 'FoodPresentation', 'FoodConcepts', 'Flavortype', 'Mealtime']
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        combined_data = pd.concat([df[column], user_preferred_df[column]])
        le.fit(combined_data)
        df[column] = le.transform(df[column])
        user_preferred_df[column] = le.transform(user_preferred_df[column])
        label_encoders[column] = le

    df['Combined_Features'] = df[categorical_columns].astype(str).agg(' '.join, axis=1)
    user_preferred_df['Combined_Features'] = user_preferred_df[categorical_columns].astype(str).agg(' '.join, axis=1)

    return df, user_preferred_df
