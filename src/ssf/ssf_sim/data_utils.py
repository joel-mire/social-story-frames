"""
Data utilities for preparing DataFrame data for SSF-Sim computation.

These are optional helper functions to convert DataFrames into the dict structures
expected by CommunitySsfSim and StorySsfSim. Users can also prepare data manually.
"""

from typing import Dict, List, Optional
import pandas as pd

# Import core utilities from generic module
from ssf.utils.TaxonomyDataUtils import (
    normalize_response,
    normalize_var_vals,
    build_sublabel_counts as _build_sublabel_counts,
    build_varvals_list as _build_varvals_list
)


def build_community_sublabel_counts(
    df: pd.DataFrame,
    taxonomy,
    groupby_col: str,
    sublabels_to_ignore: Optional[List[str]] = None
):
    if sublabels_to_ignore is None:
        sublabels_to_ignore = ['other']

    return _build_sublabel_counts(
        df=df,
        taxonomy=taxonomy,
        groupby_col=groupby_col,
        sublabels_to_ignore=sublabels_to_ignore
    )


def build_community_varvals_list(
    df: pd.DataFrame,
    taxonomy,
    groupby_col: str
):
    result, _ = _build_varvals_list(
        df=df,
        taxonomy=taxonomy,
        groupby_col=groupby_col,
        return_quality_stats=False
    )
    return result


def extract_story_sublabels(
    df: pd.DataFrame,
    story_id: str,
    taxonomy,
    story_id_col: str = 'id',
    sublabel_cols: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    row = df[df[story_id_col] == story_id]

    if row.empty:
        raise ValueError(f"Story ID '{story_id}' not found in DataFrame")

    row = row.iloc[0]

    result = {}
    for dim in taxonomy.get_dims():
        col_name = sublabel_cols.get(dim, dim) if sublabel_cols else dim
        result[dim] = row[col_name] if isinstance(row[col_name], list) else []

    return result


def extract_story_varvals(
    df: pd.DataFrame,
    story_id: str,
    taxonomy,
    story_id_col: str = 'id',
    response_cols: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    row = df[df[story_id_col] == story_id]

    if row.empty:
        raise ValueError(f"Story ID '{story_id}' not found in DataFrame")

    row = row.iloc[0]

    result = {}
    for dim in taxonomy.get_dims():
        col_name = response_cols.get(dim, dim) if response_cols else dim

        if col_name not in row.index:
            result[dim] = []
            continue

        response = row[col_name]

        # Normalize and extract
        response = normalize_response(response)
        var_vals = taxonomy.get_var_vals(dim, response)
        var_vals = normalize_var_vals(var_vals)

        # Quality checks
        if var_vals == ["ERROR"]:
            result[dim] = []
            continue

        # Check if unmodified
        dim_vars = taxonomy.get_dim_vars_dict()[dim]
        if any(var_val == dim_var for var_val, dim_var in zip(var_vals, dim_vars)):
            result[dim] = []
            continue

        result[dim] = var_vals

    return result
