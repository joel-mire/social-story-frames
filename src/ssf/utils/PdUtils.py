import pandas as pd
import random

def save_df(df, path):
  df.to_csv(path, index=False)

def remove_nan_rows(df: pd.DataFrame, cols_to_check: list) -> pd.DataFrame:
    """
    Remove rows where ALL specified columns are NaN.

    Args:
        df: DataFrame to filter
        cols_to_check: List of column names to check

    Returns:
        Filtered DataFrame with index reset
    """
    return df.dropna(subset=cols_to_check, how='all').reset_index(drop=True)

def sample_random_generation(
    df: pd.DataFrame,
    model_name: str,
    taxonomy,
    n_generations: int = 3
) -> pd.DataFrame:
    """
    For each dimension, randomly sample one generation from available options.
    
    Args:
        df: DataFrame with generation columns
        model_name: Model identifier (e.g., "gpt_4o")
        taxonomy: Taxonomy object with get_dims() method
        n_generations: Number of generation options per dimension (default: 3)
    
    Returns:
        DataFrame with added columns: {model_name}_{dim}_genRand and 
        {model_name}_{dim}_genRandIdx for each dimension
    """
    df = df.copy()
    
    for dim in taxonomy.get_dims():
        sampled_response_list = []
        sampled_response_idx_list = []
        
        for _, row in df.iterrows():
            options = {
                j: row[f"{model_name}_{dim}_gen{j}"]
                for j in range(n_generations)
                if f"{model_name}_{dim}_gen{j}" in row 
                and row[f"{model_name}_{dim}_gen{j}"] != ""
            }
            
            if options:
                random_key, random_value = random.choice(list(options.items()))
                sampled_response_list.append(random_value)
                sampled_response_idx_list.append(random_key)
            else:
                sampled_response_list.append("")
                sampled_response_idx_list.append(None)
        
        df[f"{model_name}_{dim}_genRand"] = sampled_response_list
        df[f"{model_name}_{dim}_genRandIdx"] = sampled_response_idx_list
    
    return df