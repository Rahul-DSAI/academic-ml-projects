import torch
import torch.nn.functional as F
import pandas as pd

def calculate_similarity(df, user_preferred_df):
    # Combine both dataframes to ensure consistent one-hot encoding
    combined_df = pd.concat([df, user_preferred_df])
    
    # One-hot encode the combined features
    combined_features_tensor = torch.tensor(pd.get_dummies(combined_df['Combined_Features']).values, dtype=torch.float32)
    
    # Separate the tensors back
    df_features_tensor = combined_features_tensor[:len(df)]
    user_features_tensor = combined_features_tensor[len(df):]
    
    # Calculate the mean feature vector for the user preferred foods
    mean_user_features_tensor = user_features_tensor.mean(dim=0)
    
    cosine_sim = F.cosine_similarity(df_features_tensor, mean_user_features_tensor.unsqueeze(0).repeat(df_features_tensor.size(0), 1))
    df['Similarity_Score'] = cosine_sim.detach().numpy()
    
    return df
