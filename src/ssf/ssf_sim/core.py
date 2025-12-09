from typing import Dict, Counter
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

def convert_to_prob_distribution(counts: np.ndarray) -> np.ndarray:
    if np.any(counts < 0):
        raise ValueError("Counts cannot be negative")
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def compute_js_distance(
    sublabel_support: list,
    count_dict_1: Counter,
    count_dict_2: Counter
) -> float:
    p_counts = np.array([count_dict_1.get(sublabel, 0) for sublabel in sublabel_support])
    q_counts = np.array([count_dict_2.get(sublabel, 0) for sublabel in sublabel_support])

    # Handle case where both distributions are empty (both items have no labels)
    if p_counts.sum() == 0 and q_counts.sum() == 0:
        return 0.0  # Both empty = identical

    # Handle case where only one distribution is empty
    if p_counts.sum() == 0 or q_counts.sum() == 0:
        return 1.0  # One empty, one not = completely different

    p = convert_to_prob_distribution(p_counts)
    q = convert_to_prob_distribution(q_counts)
    return jensenshannon(p, q, base=2)


def compute_cosine_similarity(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity([embedding_1], [embedding_2])[0][0]


def normalize_similarity_matrix_ranks(sim_df: pd.DataFrame) -> pd.DataFrame:
    # Create mask for upper triangle only (excluding diagonal)
    mask = np.triu(np.ones(sim_df.shape, dtype=bool), k=1)

    # Extract upper triangle values, rank them, and normalize
    upper_tri_values = sim_df.values[mask]

    # Check for NaN values and raise error if found
    if np.isnan(upper_tri_values).any():
        raise ValueError("NaN values found in upper triangle of similarity matrix")

    if len(upper_tri_values) == 0:
        raise ValueError("No similarity values found")

    ranks = pd.Series(upper_tri_values).rank(method='average')
    normalized_ranks = ranks / len(ranks)  # Normalize to [0, 1]

    # Put ranks back into matrix structure (upper triangle only)
    result = sim_df.copy()
    result.values[mask] = normalized_ranks

    # Set diagonal to NaN to exclude from analysis
    np.fill_diagonal(result.values, np.nan)

    return result

def weighted_borda_fusion(
    sim_df_1: pd.DataFrame,
    sim_df_2: pd.DataFrame,
    lambda_param: float
) -> pd.DataFrame:
    # Normalize both matrices to ranks
    sim_df_1_norm = normalize_similarity_matrix_ranks(sim_df_1)
    sim_df_2_norm = normalize_similarity_matrix_ranks(sim_df_2)

    # Weighted combination
    result = lambda_param * sim_df_1_norm + (1 - lambda_param) * sim_df_2_norm

    # Ensure diagonal remains NaN after combination
    np.fill_diagonal(result.values, np.nan)

    return result
