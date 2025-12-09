"""
Utilities for computing Normalized Pointwise Mutual Information (NPMI).

NPMI measures the association between categorical variables, ranging from -1 (never co-occur)
to 1 (always co-occur together), with 0 indicating independence.

This module provides functions for computing NPMI on taxonomy-annotated data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def compute_npmi(
    df: pd.DataFrame,
    dim_x: str,
    dim_y: str,
    subdims: Dict[str, List[str]],
    min_support: int = 10
) -> pd.DataFrame:
    """
    Compute Normalized Pointwise Mutual Information (NPMI) between sub-dimensions of dim_x and dim_y.

    NPMI measures the association between categorical variables, ranging from -1 (never co-occur)
    to 1 (always co-occur together), with 0 indicating independence.

    Args:
        df: DataFrame with taxonomy annotations (must have {dim}_cats columns)
        dim_x: First dimension name
        dim_y: Second dimension name
        subdims: Dictionary mapping dimension names to their valid subcategories
        min_support: Minimum co-occurrence count threshold. Cells with joint count < min_support
                    are set to NaN to avoid spurious associations from rare events.

    Returns:
        DataFrame with NPMI values between subcategories of dim_x (rows) and dim_y (columns).
        Values range from -1 to 1, with NaN for low-support pairs.

    Example:
        >>> from ssf.utils.NpmiUtils import compute_npmi
        >>> subdims = {dim: list(taxonomy.dim_data_dict[dim].keys()) for dim in taxonomy.get_dims()}
        >>> npmi_matrix = compute_npmi(
        ...     df=tc_analysis_df,
        ...     dim_x='overall_goal',
        ...     dim_y='narrative_intent',
        ...     subdims=subdims,
        ...     min_support=10
        ... )
        >>> # Use with visualization
        >>> ResultsVisUtils.plot_npmi_heatmap(npmi_matrix, "overall_goal$narrative_intent")

    Notes:
        - Both dimension columns must contain lists of string values
        - Self-associations (same category appearing twice) are excluded when dim_x == dim_y
        - Only categories that appear in both subdims and actual data are included
    """
    def get_valid_categories(dim, df, subdims):
        """Get categories that exist in both subdims definition and actual data."""
        # Get all categories that appear in the data for this dimension
        data_cats = set()
        for _, row in df.iterrows():
            data_cats.update(row[f"{dim}_cats"])

        # Only keep categories that are both in subdims and in data
        valid_cats = [cat for cat in subdims[dim] if cat in data_cats]
        return valid_cats

    valid_cats_x = get_valid_categories(dim_x, df, subdims)
    valid_cats_y = get_valid_categories(dim_y, df, subdims)

    # Compute joint counts (co-occurrence matrix)
    cooccurrence_counts = pd.DataFrame(0, index=valid_cats_x, columns=valid_cats_y)
    for _, row in df.iterrows():
        for cat1 in set(row[f"{dim_x}_cats"]):
            if cat1 not in valid_cats_x:
                continue
            for cat2 in set(row[f"{dim_y}_cats"]):
                if cat2 not in valid_cats_y:
                    continue
                # Exclude self-associations within same dimension
                if dim_x == dim_y and cat1 == cat2:
                    continue
                cooccurrence_counts.loc[cat1, cat2] += 1

    # Compute marginal counts for dim_x
    dim_x_subdims_counts = pd.Series(0, index=valid_cats_x)
    for _, row in df.iterrows():
        for cat in set(row[f"{dim_x}_cats"]):
            if cat in valid_cats_x:
                dim_x_subdims_counts[cat] += 1

    # Compute marginal counts for dim_y
    dim_y_subdims_counts = pd.Series(0, index=valid_cats_y)
    for _, row in df.iterrows():
        for cat in set(row[f"{dim_y}_cats"]):
            if cat in valid_cats_y:
                dim_y_subdims_counts[cat] += 1

    # Convert counts to probabilities
    total_stories = len(df)
    p_x = dim_x_subdims_counts / total_stories
    p_y = dim_y_subdims_counts / total_stories

    # Joint probabilities
    p_xy = cooccurrence_counts / total_stories

    # Compute expected probabilities under independence
    pxpy = np.outer(p_x.values, p_y.values)
    pxy = p_xy.values

    # Compute PMI and normalize to NPMI
    # PMI(x,y) = log2(P(x,y) / (P(x) * P(y)))
    # NPMI(x,y) = PMI(x,y) / -log2(P(x,y))
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2(np.where(pxy > 0, pxy / pxpy, 1))
        npmi = np.where(pxy > 0, pmi / (-np.log2(pxy)), 0)

    result = pd.DataFrame(npmi, index=valid_cats_x, columns=valid_cats_y)

    # Mask low support cells to avoid spurious associations
    mask = cooccurrence_counts < min_support
    result = result.mask(mask)

    return result