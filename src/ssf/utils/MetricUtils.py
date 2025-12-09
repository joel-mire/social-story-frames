import numpy as np
import pandas as pd

def get_top_rank_changes(df1, df2, metric1_name, metric2_name, top_k=10):
    """
    Generic utility to find top k pairs with largest rank changes between two similarity matrices.
    
    Args:
        df1: First similarity DataFrame (square matrix)
        df2: Second similarity DataFrame (square matrix)
        metric1_name: Name for first similarity metric
        metric2_name: Name for second similarity metric
        top_k: Number of top changes to return
    
    Returns:
        DataFrame with top k pairs sorted by absolute rank gap
    """
    # Extract upper triangular values (excluding diagonal)
    triu_indices = np.triu_indices_from(df1, k=1)
    sim1_flat = df1.values[triu_indices]
    sim2_flat = df2.values[triu_indices]
    
    # Get subreddit names for pair identification
    subreddit_names = df1.index.tolist()
    pairs = [(subreddit_names[i], subreddit_names[j]) for i, j in zip(*triu_indices)]
    
    # Create dataframe with similarity values
    pair_df = pd.DataFrame({
        'pair': pairs,
        f'{metric1_name.lower().replace(" ", "_")}_sim': sim1_flat,
        f'{metric2_name.lower().replace(" ", "_")}_sim': sim2_flat
    })
    
    # Calculate ranks (higher similarity = lower rank number)
    pair_df[f'{metric1_name.lower().replace(" ", "_")}_rank'] = pair_df[f'{metric1_name.lower().replace(" ", "_")}_sim'].rank(ascending=False)
    pair_df[f'{metric2_name.lower().replace(" ", "_")}_rank'] = pair_df[f'{metric2_name.lower().replace(" ", "_")}_sim'].rank(ascending=False)
    
    # Calculate rank differences (positive = metric1 ranks higher than metric2)
    pair_df['rank_gap_signed'] = pair_df[f'{metric2_name.lower().replace(" ", "_")}_rank'] - pair_df[f'{metric1_name.lower().replace(" ", "_")}_rank']
    pair_df['rank_gap_abs'] = pair_df['rank_gap_signed'].abs()
    
    # Return top k pairs sorted by absolute rank gap
    return pair_df.nlargest(top_k, 'rank_gap_abs')

def get_agreement_cases(df1, df2, metric1_name, metric2_name, top_k=3):
    """
    Get cases where both metrics strongly agree on high similarity or low similarity.
    
    Args:
        df1: First similarity DataFrame (square matrix)
        df2: Second similarity DataFrame (square matrix)
        metric1_name: Name for first similarity metric
        metric2_name: Name for second similarity metric
        top_k: Number of agreement cases to return for each type
    
    Returns:
        tuple: (high_similarity_agreement, low_similarity_agreement)
    """
    # Extract upper triangular values (excluding diagonal)
    triu_indices = np.triu_indices_from(df1, k=1)
    sim1_flat = df1.values[triu_indices]
    sim2_flat = df2.values[triu_indices]
    
    # Get subreddit names for pair identification
    subreddit_names = df1.index.tolist()
    pairs = [(subreddit_names[i], subreddit_names[j]) for i, j in zip(*triu_indices)]
    
    # Create dataframe with similarity values
    pair_df = pd.DataFrame({
        'pair': pairs,
        f'{metric1_name.lower().replace(" ", "_")}_sim': sim1_flat,
        f'{metric2_name.lower().replace(" ", "_")}_sim': sim2_flat
    })
    
    # Calculate ranks (higher similarity = lower rank number)
    pair_df[f'{metric1_name.lower().replace(" ", "_")}_rank'] = pair_df[f'{metric1_name.lower().replace(" ", "_")}_sim'].rank(ascending=False)
    pair_df[f'{metric2_name.lower().replace(" ", "_")}_rank'] = pair_df[f'{metric2_name.lower().replace(" ", "_")}_sim'].rank(ascending=False)
    
    # Calculate rank differences (positive = metric1 ranks higher than metric2)
    pair_df['rank_gap_signed'] = pair_df[f'{metric2_name.lower().replace(" ", "_")}_rank'] - pair_df[f'{metric1_name.lower().replace(" ", "_")}_rank']
    pair_df['rank_gap_abs'] = pair_df['rank_gap_signed'].abs()
    
    # Calculate average rank for agreement detection
    pair_df['avg_rank'] = (pair_df[f'{metric1_name.lower().replace(" ", "_")}_rank'] + 
                          pair_df[f'{metric2_name.lower().replace(" ", "_")}_rank']) / 2
    
    # Get high similarity agreement (both metrics rank pairs highly, small rank gap)
    high_sim_candidates = pair_df[pair_df['avg_rank'] <= 50]  # Top 50 average rank
    high_similarity_agreement = high_sim_candidates.nsmallest(top_k, 'rank_gap_abs')
    
    # Get low similarity agreement (both metrics rank pairs lowly, small rank gap)
    low_sim_candidates = pair_df[pair_df['avg_rank'] >= len(pair_df) - 50]  # Bottom 50 average rank
    low_similarity_agreement = low_sim_candidates.nsmallest(top_k, 'rank_gap_abs')
    
    return high_similarity_agreement, low_similarity_agreement

def get_top_positive_negative_changes(df1, df2, metric1_name, metric2_name, top_k=5):
    """
    Get top k positive and top k negative rank changes separately.
    
    Args:
        df1: First similarity DataFrame (square matrix)
        df2: Second similarity DataFrame (square matrix)
        metric1_name: Name for first similarity metric
        metric2_name: Name for second similarity metric
        top_k: Number of top positive and negative changes to return
    
    Returns:
        tuple: (top_positive_changes, top_negative_changes)
    """
    # Extract upper triangular values (excluding diagonal)
    triu_indices = np.triu_indices_from(df1, k=1)
    sim1_flat = df1.values[triu_indices]
    sim2_flat = df2.values[triu_indices]
    
    # Get subreddit names for pair identification
    subreddit_names = df1.index.tolist()
    pairs = [(subreddit_names[i], subreddit_names[j]) for i, j in zip(*triu_indices)]
    
    # Create dataframe with similarity values
    pair_df = pd.DataFrame({
        'pair': pairs,
        f'{metric1_name.lower().replace(" ", "_")}_sim': sim1_flat,
        f'{metric2_name.lower().replace(" ", "_")}_sim': sim2_flat
    })
    
    # Calculate ranks (higher similarity = lower rank number)
    pair_df[f'{metric1_name.lower().replace(" ", "_")}_rank'] = pair_df[f'{metric1_name.lower().replace(" ", "_")}_sim'].rank(ascending=False)
    pair_df[f'{metric2_name.lower().replace(" ", "_")}_rank'] = pair_df[f'{metric2_name.lower().replace(" ", "_")}_sim'].rank(ascending=False)
    
    # Calculate rank differences (positive = metric1 ranks higher than metric2)
    pair_df['rank_gap_signed'] = pair_df[f'{metric2_name.lower().replace(" ", "_")}_rank'] - pair_df[f'{metric1_name.lower().replace(" ", "_")}_rank']
    pair_df['rank_gap_abs'] = pair_df['rank_gap_signed'].abs()
    
    # Get top positive changes (metric1 ranks much higher than metric2)
    top_positive = pair_df[pair_df['rank_gap_signed'] > 0].nlargest(top_k, 'rank_gap_signed')
    
    # Get top negative changes (metric2 ranks much higher than metric1)
    top_negative = pair_df[pair_df['rank_gap_signed'] < 0].nsmallest(top_k, 'rank_gap_signed')
    
    return top_positive, top_negative
