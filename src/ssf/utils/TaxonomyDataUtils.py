"""
Data utilities for working with Taxonomy-annotated DataFrames.

This module provides generic utilities for extracting and processing taxonomy data
from DataFrames. These utilities are used by both the ssf_sim package and analysis
notebooks.
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from collections import Counter
from ssf.helpers import map_moral_values

def normalize_response(response: str) -> str:
    """
    Add period if missing from response.

    Args:
        response: Generated response string

    Returns:
        Response with trailing period
    """
    if not response.strip().endswith("."):
        response += "."
    return response


def normalize_var_vals(var_vals: List[str]) -> List[str]:
    """
    Remove {{ }} template artifacts from variable values.

    Args:
        var_vals: List of variable values extracted from template

    Returns:
        Cleaned variable values
    """
    result = []
    for v in var_vals:
        # Step 1: If starts with "{{", take everything after
        if v.startswith('{{'):
            v = v[2:]
        # Step 2: If ends with "}}", take everything before
        if v.endswith('}}'):
            v = v[:-2]
        # Step 3: If "}}" still exists anywhere, take everything after it
        if '}}' in v:
            v = v[v.find('}}') + 2:]

        result.append(v)
    return result


def build_sublabel_counts(
    df: pd.DataFrame,
    taxonomy,
    groupby_col: Optional[str] = None,
    sublabels_to_ignore: Optional[List[str]] = None
) -> Dict:
    """
    Build sublabel counts from DataFrame, with flexible aggregation modes.

    Expects DataFrame to have columns named: {dim}_cats for each dimension.

    Args:
        df: DataFrame with rows=stories, columns include {dim}_cats with lists of sublabels
        taxonomy: Taxonomy instance
        groupby_col: Optional column to group by (e.g., 'subreddit', 'id').
                    If None, aggregates across all rows by dimension only.
        sublabels_to_ignore: Sublabels to exclude from counts (e.g., ['other'])

    Returns:
        If groupby_col is specified:
            {
                'item1': {
                    'dim1': Counter({'sublabel1': 10, 'sublabel2': 5, ...}),
                    'dim2': Counter({...}),
                    ...
                },
                'item2': {...},
                ...
            }
        If groupby_col is None:
            {
                'dim1': Counter({'sublabel1': 10, 'sublabel2': 5, ...}),
                'dim2': Counter({...}),
                ...
            }

    Example:
        >>> # Community-level aggregation
        >>> sublabel_counts = build_sublabel_counts(
        ...     df=tc_analysis_df,
        ...     taxonomy=taxonomy,
        ...     groupby_col='subreddit',
        ...     sublabels_to_ignore=['other']
        ... )

        >>> # Dimension-level aggregation (no grouping)
        >>> dim_counts = build_sublabel_counts(
        ...     df=tc_analysis_df,
        ...     taxonomy=taxonomy,
        ...     sublabels_to_ignore=['other']
        ... )
    """
    if sublabels_to_ignore is None:
        sublabels_to_ignore = ['other']

    if groupby_col is not None:
        # Group-level aggregation
        result = {}
        groups = df.groupby(groupby_col)

        for item_name, item_df in groups:
            dim_counts = {}

            for dim in taxonomy.get_dims():
                col_name = f"{dim}_cats"

                # Explode lists and count
                exploded = item_df[col_name].explode()
                filtered = exploded[~exploded.isin(sublabels_to_ignore)]
                dim_counts[dim] = Counter(filtered)

            result[item_name] = dim_counts

        return result
    else:
        # Dimension-level aggregation (no grouping)
        result = {}

        for dim in taxonomy.get_dims():
            col_name = f"{dim}_cats"

            # Explode lists and count
            exploded = df[col_name].explode()
            filtered = exploded[~exploded.isin(sublabels_to_ignore)]
            result[dim] = Counter(filtered)

        return result


def build_varvals_list(
    df: pd.DataFrame,
    taxonomy,
    groupby_col: Optional[str] = None,
    instance_id_col: Optional[str] = None,
    return_quality_stats: bool = False
) -> Tuple[Dict, Optional[Dict]]:
    """
    Build variable values list from DataFrame with flexible aggregation modes.

    This function extracts variable values from template-filled response columns
    and groups them according to the specified mode.

    Expects DataFrame to have columns named: {dim}_gen for each dimension.

    Args:
        df: DataFrame with stories containing {dim}_gen columns
        taxonomy: Taxonomy instance
        groupby_col: Optional column to group by (e.g., 'subreddit').
                    Returns {group: {dim: [[varvals]]}} if specified.
        instance_id_col: Optional column for instance IDs.
                        Returns {(id, dim): [[varvals]]} if specified.
        return_quality_stats: If True, return quality statistics as second element of tuple

    Returns:
        Tuple of (varvals_dict, quality_stats_dict or None)

        If groupby_col is specified (community mode):
            {
                'item1': {
                    'dim1': [
                        ['var1_val', 'var2_val'],  # story 1 variable values
                        ['var1_val', 'var2_val'],  # story 2 variable values
                        ...
                    ],
                    'dim2': [...],
                    ...
                },
                'item2': {...},
                ...
            }

        If instance_id_col is specified (instance mode):
            {
                ('id1', 'dim1'): [['var1_val', 'var2_val'], ...],
                ('id1', 'dim2'): [['var1_val', 'var2_val'], ...],
                ('id2', 'dim1'): [...],
                ...
            }

        If neither is specified (dimension mode):
            {
                'dim1': [['var1_val', 'var2_val'], ...],
                'dim2': [['var1_val', 'var2_val'], ...],
                ...
            }

        quality_stats_dict (if return_quality_stats=True):
            {
                'invalid_fmt rate': float,
                'unmodified rate': float,
                'quality rate': float
            }

    Example:
        >>> # Community-level aggregation
        >>> varvals_list, _ = build_varvals_list(
        ...     df=tc_analysis_df,
        ...     taxonomy=taxonomy,
        ...     groupby_col='subreddit'
        ... )

        >>> # Instance-level (for metrics)
        >>> instance_varvals, stats = build_varvals_list(
        ...     df=stories_df,
        ...     taxonomy=taxonomy,
        ...     instance_id_col='id',
        ...     return_quality_stats=True
        ... )

        >>> # Dimension-level (no grouping)
        >>> dim_varvals, _ = build_varvals_list(
        ...     df=stories_df,
        ...     taxonomy=taxonomy
        ... )
    """
    invalid_fmt_count = 0
    unmodified_count = 0
    total_count = 0

    # Determine the aggregation mode and initialize result structure
    if groupby_col is not None:
        # Community mode: {group: {dim: [[varvals]]}}
        result = defaultdict(lambda: defaultdict(list))
    elif instance_id_col is not None:
        # Instance mode: {(id, dim): [[varvals]]}
        result = defaultdict(list)
    else:
        # Dimension mode: {dim: [[varvals]]}
        result = defaultdict(list)

    for _, row in df.iterrows():
        for dim in taxonomy.get_dims():
            col_name = f"{dim}_gen"

            if col_name not in row.index:
                continue

            response = row[col_name]

            # Normalize and extract
            response = normalize_response(response)
            var_vals = taxonomy.get_var_vals(dim, response)
            var_vals = normalize_var_vals(var_vals)

            total_count += 1

            # Quality checks
            if var_vals == ["ERROR"]:
                invalid_fmt_count += 1
                continue  # Skip invalid

            # Check if unmodified (same as template variables)
            dim_vars = taxonomy.get_dim_vars_dict()[dim]
            if any(var_val == dim_var for var_val, dim_var in zip(var_vals, dim_vars)):
                unmodified_count += 1
                continue  # Skip unmodified

            # Add to result based on mode
            if groupby_col is not None:
                # Community mode
                group_name = row[groupby_col]
                result[group_name][dim].append(var_vals)
            elif instance_id_col is not None:
                # Instance mode
                instance_id = row[instance_id_col]
                key = (instance_id, dim)
                result[key].append(var_vals)
            else:
                # Dimension mode
                result[dim].append(var_vals)

    # Convert defaultdicts to regular dicts
    if groupby_col is not None:
        result = {k: dict(v) for k, v in result.items()}
    else:
        result = dict(result)

    # Calculate quality stats if requested
    quality_stats = None
    if return_quality_stats:
        if total_count == 0:
            quality_stats = {
                "invalid_fmt rate": 0,
                "unmodified rate": 0,
                "quality rate": 0
            }
        else:
            quality_stats = {
                "invalid_fmt rate": invalid_fmt_count / total_count,
                "unmodified rate": unmodified_count / total_count,
                "quality rate": (total_count - invalid_fmt_count - unmodified_count) / total_count
            }

    return result, quality_stats


def prepare_dataframe_for_varvals(
    df: pd.DataFrame,
    taxonomy,
    response_col_prefix: str,
    response_col_suffix: str,
    response_col_is_only_suffix: bool = False
) -> pd.DataFrame:
    """
    Prepare a DataFrame by creating standardized {dim}_gen columns from custom column formats.

    This is useful for converting DataFrames with non-standard column naming (e.g., from
    different data sources or annotation tools) into the standard format expected by
    build_varvals_list.

    Args:
        df: Input DataFrame
        taxonomy: Taxonomy instance
        response_col_prefix: Prefix for response columns (e.g., "gpt_4o_2024_11_20_")
        response_col_suffix: Suffix for response columns (e.g., "_gen0" or ":::text_box")
        response_col_is_only_suffix: If True, column contains only suffix (POTATO format)

    Returns:
        DataFrame with new {dim}_gen columns added

    Example:
        >>> # Convert GPT-4 format to standard format
        >>> df_prepared = prepare_dataframe_for_varvals(
        ...     df=raw_df,
        ...     taxonomy=taxonomy,
        ...     response_col_prefix="gpt_4o_2024_11_20_",
        ...     response_col_suffix="_gen0"
        ... )
        >>> # Now df_prepared has columns like "overall_goal_gen", "narrative_intent_gen", etc.

        >>> # Convert POTATO format
        >>> hw_prepared = prepare_dataframe_for_varvals(
        ...     df=hw_df,
        ...     taxonomy=taxonomy,
        ...     response_col_prefix="",
        ...     response_col_suffix=":::text_box",
        ...     response_col_is_only_suffix=True
        ... )
    """
    df = df.copy()

    for dim in taxonomy.get_dims():
        source_col = f"{response_col_prefix}{dim}{response_col_suffix}"
        target_col = f"{dim}_gen"

        if source_col not in df.columns:
            continue

        if response_col_is_only_suffix:
            # POTATO format: need to prepend template prefix
            template_prefix = taxonomy.get_template_prefix(dim)
            df[target_col] = df[source_col].apply(
                lambda x: f"{template_prefix}{x}" if pd.notna(x) else x
            )
        else:
            # Standard format: just copy/rename
            df[target_col] = df[source_col]

    return df


def get_label_counts(df: pd.DataFrame, column_name: str, filter_col: str = None, filter_value: bool = None):
    """
    Count occurrences of each label in a multilabel column (where each cell is a list of labels).

    Args:
        df: DataFrame to analyze
        column_name: Name of column containing lists of labels
        filter_col: Optional column name to filter by
        filter_value: Optional value to filter on

    Returns:
        Series with counts for each label, sorted descending

    Example:
        >>> # Count sublabels in overall_goal_cats column
        >>> counts = get_label_counts(df, 'overall_goal_cats')

        >>> # Count only for stratified subset
        >>> stratified_counts = get_label_counts(df, 'overall_goal_cats',
        ...                                      filter_col='is_sampled_for_subreddit',
        ...                                      filter_value=True)
    """

    # Apply filter if specified
    if filter_col is not None and filter_value is not None:
        df_filtered = df[df[filter_col] == filter_value]
    else:
        df_filtered = df

    # Flatten the list of lists
    all_labels = [label for sublist in df_filtered[column_name] for label in sublist]

    # Count occurrences
    counts = Counter(all_labels)

    # Convert to Series for consistency
    return pd.Series(counts).sort_values(ascending=False)


def get_label_proportions(df: pd.DataFrame, column_name: str, filter_col: str = None, filter_value: bool = None):
    """
    Calculate proportions of each label in a multilabel column (where each cell is a list of labels).

    Args:
        df: DataFrame to analyze
        column_name: Name of column containing lists of labels
        filter_col: Optional column name to filter by
        filter_value: Optional value to filter on

    Returns:
        Series with proportions for each label

    Example:
        >>> # Get proportions of sublabels
        >>> proportions = get_label_proportions(df, 'overall_goal_cats')
    """
    counts = get_label_counts(df, column_name, filter_col, filter_value)
    total = counts.sum()
    return counts / total if total > 0 else counts


def get_sublabel_support(taxonomy, dim: str, allow_other: bool = True) -> List[str]:
    """
    Get the valid sublabels for a dimension, excluding excluded categories.

    Args:
        taxonomy: Taxonomy instance
        dim: Dimension name (e.g., 'overall_goal', 'moral')
        allow_other: If False, exclude 'other' from sublabels

    Returns:
        List of valid sublabel strings

    Example:
        >>> sublabels = get_sublabel_support(taxonomy, 'overall_goal', allow_other=False)
        >>> # ['inform', 'persuade', 'entertain', ...]
    """

    sublabels = [
        sublabel for sublabel in taxonomy.dim_data_dict[dim].keys()
        if sublabel not in taxonomy.get_excluded_categories(dim)
    ]
    sublabels = list(set(sublabels))  # Remove duplicates

    if not allow_other and 'other' in sublabels:
        sublabels.remove('other')

    if dim == 'moral':
        sublabels = map_moral_values(sublabels)

    return sublabels


def filter_sublabels(sublabels: List[str], sublabel_support: List[str]) -> List[str]:
    """
    Filter a list of sublabels to only include those in the support set.

    Args:
        sublabels: List of sublabels to filter
        sublabel_support: List of valid sublabels

    Returns:
        Filtered list of sublabels

    Example:
        >>> support = get_sublabel_support(taxonomy, 'overall_goal')
        >>> filtered = filter_sublabels(['inform', 'invalid_label'], support)
        >>> # ['inform']
    """
    return [sublabel for sublabel in sublabels if sublabel in sublabel_support]


def canonicalize_cats(tc_analysis_df: pd.DataFrame, taxonomy) -> pd.DataFrame:
    """
    Canonicalize category labels in a DataFrame by filtering excluded categories
    and applying dimension-specific mappings.

    For the moral dimension, this applies moral value mapping to simplify categories.
    For other dimensions, it filters to valid sublabels only.

    Expects columns named: {dim}_cats for each dimension.

    Args:
        tc_analysis_df: DataFrame with {dim}_cats columns containing lists of categories
        taxonomy: Taxonomy instance

    Returns:
        DataFrame with canonicalized category columns

    Example:
        >>> df = canonicalize_cats(tc_analysis_df, taxonomy)
        >>> # moral_cats column now has mapped values (e.g., 'care' instead of 'harm/care')
    """
    for dim in taxonomy.get_dims():
        cat_col = f"{dim}_cats"

        if cat_col not in tc_analysis_df.columns:
            continue

        if dim == 'moral':
            # For moral dimension, apply mapping to the data first, then filter
            def process_moral_cats(cats):
                # First filter out excluded categories from raw data
                filtered_cats = [
                    cat for cat in cats
                    if cat in taxonomy.dim_data_dict[dim]
                    and cat not in taxonomy.get_excluded_categories(dim)
                ]
                # Then apply moral mapping to get simplified categories
                return map_moral_values(filtered_cats)

            tc_analysis_df[cat_col] = tc_analysis_df[cat_col].apply(process_moral_cats)
        else:
            # For other dimensions, use the helper functions
            sublabel_support = get_sublabel_support(taxonomy, dim, allow_other=True)
            tc_analysis_df[cat_col] = tc_analysis_df[cat_col].apply(
                lambda cats: filter_sublabels(cats, sublabel_support)
            )

    return tc_analysis_df


def get_cats_col_name(prompt_col_suffix: str, dim: str, genIdxStr: str) -> str:
    """
    Get the column name for category predictions in the standard format.

    Args:
        prompt_col_suffix: Prompt suffix (e.g., 'prompt_default')
        dim: Dimension name (e.g., 'overall_goal')
        genIdxStr: Generation index string (e.g., 'gen0')

    Returns:
        Column name string

    Example:
        >>> col = get_cats_col_name('prompt_default', 'overall_goal', 'gen0')
        >>> # 'prompt_default$overall_goal_gen0_cats'
    """
    return f"{prompt_col_suffix}${dim}_{genIdxStr}_cats"


def add_is_in_ml_df_col(full_df: pd.DataFrame, ml_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean column indicating whether each story is in the ML dataset.

    Args:
        full_df: Full DataFrame with all stories
        ml_df: ML DataFrame subset

    Returns:
        full_df with 'is_in_ml_df' column added

    Example:
        >>> full_df = add_is_in_ml_df_col(full_df, ml_df)
        >>> print(full_df['is_in_ml_df'].sum())  # Number of ML stories
    """
    if not ('id' in full_df.columns and 'id' in ml_df.columns):
        raise ValueError("Both dataframes must have an 'id' column for the instance id")

    ml_df_ids = set(ml_df['id'])
    full_df['is_in_ml_df'] = full_df['id'].isin(ml_df_ids)

    print(f"\nAdded is_in_ml_df column. {full_df['is_in_ml_df'].sum()} out of {len(full_df)} stories are in ML dataset")

    return full_df


def get_tc_analysis_df(
    full_df: pd.DataFrame,
    taxonomy,
    prompt_col_suffix: str = "prompt_default",
    genIdxStr: str = "gen0",
    min_subreddit_count: int = 45,
    sample_size_per_subreddit: int = 45,
    random_seed: int = 25
) -> pd.DataFrame:
    """
    Build a taxonomy classification analysis DataFrame from the full stories DataFrame.

    This function:
    1. Extracts relevant columns (id, subreddit, text, {dim}_cats, {dim}_gen)
    2. Renames columns to standard format ({dim}_cats, {dim}_gen)
    3. Canonicalizes category labels
    4. Adds metadata columns (have_enough_subreddit_data, is_sampled_for_subreddit)
    5. Samples stories stratified by subreddit

    Args:
        full_df: Full DataFrame with story data
        taxonomy: Taxonomy instance
        prompt_col_suffix: Prefix for prompt columns (default: 'prompt_default')
        genIdxStr: Generation index string (default: 'gen0')
        min_subreddit_count: Minimum stories per subreddit to be eligible for sampling
        sample_size_per_subreddit: Number of stories to sample per eligible subreddit
        random_seed: Random seed for reproducible sampling

    Returns:
        DataFrame with columns:
        - id: Story ID
        - subreddit: Subreddit name
        - text: Story text
        - is_in_ml_df: Whether story is in ML dataset (if present in full_df)
        - split: Data split (if present in full_df)
        - {dim}_cats: Category labels for each dimension
        - {dim}_gen: Generated text for each dimension
        - have_enough_subreddit_data: Boolean flag for subreddits with >= min_subreddit_count
        - is_sampled_for_subreddit: Boolean flag for sampled stories

    Example:
        >>> tc_analysis_df = get_tc_analysis_df(
        ...     full_df=full_df,
        ...     taxonomy=taxonomy,
        ...     min_subreddit_count=45,
        ...     sample_size_per_subreddit=45
        ... )
    """
    print(f"Building tc_analysis_df from full_df with {len(full_df)} stories...")
    print(f"Full DataFrame columns: {full_df.columns.tolist()}")

    # Build column lists
    cats_cols = [get_cats_col_name(prompt_col_suffix, dim, genIdxStr) for dim in taxonomy.get_dims()]
    gen_cols = [f"{prompt_col_suffix}${dim}_{genIdxStr}" for dim in taxonomy.get_dims()]

    # Base columns to keep
    base_cols = ['id', 'meta.subreddit', '_text']

    # Optional columns (add if they exist)
    optional_cols = ['is_in_ml_df', 'split']
    cols_to_keep = base_cols + [col for col in optional_cols if col in full_df.columns] + cats_cols + gen_cols

    # Build column renaming dictionary
    oldCol_newCol_dict = {
        'meta.subreddit': 'subreddit',
        '_text': 'text',
        **{get_cats_col_name(prompt_col_suffix, dim, genIdxStr): f"{dim}_cats"
           for dim in taxonomy.get_dims()},
        **{f"{prompt_col_suffix}${dim}_{genIdxStr}": f"{dim}_gen"
           for dim in taxonomy.get_dims()}
    }

    # Extract and rename columns
    tc_analysis_df = full_df[cols_to_keep].rename(columns=oldCol_newCol_dict)

    # Canonicalize categories
    tc_analysis_df = canonicalize_cats(tc_analysis_df, taxonomy)

    # Add 'have_enough_subreddit_data' column
    subreddit_counts = tc_analysis_df['subreddit'].value_counts()
    tc_analysis_df['have_enough_subreddit_data'] = tc_analysis_df['subreddit'].map(
        lambda x: subreddit_counts[x] >= min_subreddit_count
    )

    # Stratified sampling by subreddit
    np.random.seed(random_seed)

    # Initialize the flag column to False
    tc_analysis_df['is_sampled_for_subreddit'] = False

    # Get subreddits with enough data
    eligible_subreddits = tc_analysis_df[
        tc_analysis_df['have_enough_subreddit_data'] == True
    ]['subreddit'].unique()

    print(f"\nFound {len(eligible_subreddits)} subreddits with >= {min_subreddit_count} stories")

    # For each eligible subreddit, sample stories
    for subreddit in eligible_subreddits:
        subreddit_indices = tc_analysis_df[tc_analysis_df['subreddit'] == subreddit].index
        sampled_indices = np.random.choice(subreddit_indices, size=sample_size_per_subreddit, replace=False)
        tc_analysis_df.loc[sampled_indices, 'is_sampled_for_subreddit'] = True

    print(f"Sampled {tc_analysis_df['is_sampled_for_subreddit'].sum()} stories "
          f"({sample_size_per_subreddit} per eligible subreddit)")
    print(f"\nFinal tc_analysis_df shape: {tc_analysis_df.shape}")

    return tc_analysis_df
