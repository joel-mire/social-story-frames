import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from ast import literal_eval
import math
from ssf.Taxonomy import Taxonomy
from ssf.Constants import *
from ssf.utils.TaxonomyDataUtils import get_cats_col_name
import os
from ssf.community_preprocessors.SubredditPreprocessor import SubredditPreprocessor
from ssf.Configs import load_config
from ssf.generation_strategies.configs import ModelConfig, GenerationConfig
from ssf.generation_strategies import OpenaiGenerationStrategy
from adjustText import adjust_text
from scipy import stats
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

# Dimension groupings for analysis
DIMENSION_GROUPS = {
    'author_centric': ['overall_goal', 'narrative_intent', 'author_emotional_response'],
    'reader_centric': ['causal_explanation', 'prediction', 'character_appraisal', 'moral', 'stance', 'narrative_feeling', 'aesthetic_feeling']
}

def get_cats_converters(taxonomy):
    """Get converters for category list columns."""
    return {
        get_cats_col_name(PROMPT_COL_SUFFIX_FULL_CONTEXT, dim, "gen0"): literal_eval
        for dim in taxonomy.get_dims()
    }

def get_unique_categories(stories_df, dim):
    """Get all unique categories for a dimension across the entire dataset."""
    col_name = f'prompt_default${dim}_gen0_cats'
    all_labels = []
    for val in stories_df[col_name].dropna():
        # Lists are already parsed via converters
        if isinstance(val, list):
            all_labels.extend(val)
    return set(all_labels)

def compute_subreddit_label_distribution(stories_df, subreddit, dim, col_name):
    """
    Compute probability distribution of labels for a given subreddit and dimension.

    Args:
        stories_df: DataFrame with stories
        subreddit: subreddit name
        dim: taxonomy dimension
        col_name: column name for the dimension's categories

    Returns:
        dict: probability distribution over categories
    """
    subreddit_stories = stories_df[stories_df['meta.subreddit'] == subreddit]

    # Aggregate all labels across stories in this subreddit
    all_labels = []
    for _, row in subreddit_stories.iterrows():
        labels = row[col_name]
        # Lists are already parsed via converters
        if isinstance(labels, list):
            all_labels.extend(labels)

    if not all_labels:
        return {}

    # Count labels and convert to probability distribution
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    label_probs = {label: count / total for label, count in label_counts.items()}

    return label_probs

def compute_entropy_from_distribution(prob_dist):
    """Compute Shannon entropy from probability distribution."""
    if not prob_dist:
        return np.nan

    probs = list(prob_dist.values())
    return entropy(probs, base=2)  # Using base 2 for bits

def compute_normalized_entropy(prob_dist, num_categories):
    """
    Compute normalized entropy (efficiency).

    Args:
        prob_dist: probability distribution
        num_categories: total number of possible categories

    Returns:
        float: normalized entropy in [0, 1]
    """
    if not prob_dist or num_categories <= 1:
        return np.nan

    raw_entropy = compute_entropy_from_distribution(prob_dist)
    max_entropy = math.log2(num_categories)

    return raw_entropy / max_entropy if max_entropy > 0 else np.nan

def analyze_subreddit_entropy(stories_path, use_stratified=True, min_stories=45, sample_size=45):
    """
    Compute normalized entropy of taxonomy label distributions per subreddit.

    Args:
        stories_path: path to ssf.csv
        taxonomy_dir: path to taxonomy directory
        use_stratified: if True, use stratified sampling (default: True)
        min_stories: minimum number of stories required for a subreddit to be included (default: 45)
        sample_size: number of stories to sample per subreddit (default: 45)

    Returns:
        DataFrame with subreddit normalized entropy scores
    """
    # Load taxonomy first to get converters
    taxonomy = Taxonomy(taxonomy_dir=TAXONOMY_DIR)
    dims = taxonomy.get_dims()

    # Load data with converters to parse list columns
    stories_df = pd.read_csv(stories_path, converters=get_cats_converters(taxonomy))
    print(f"Loaded {len(stories_df)} total stories")

    # Apply stratified sampling if requested
    if use_stratified:
        print(f"\nApplying stratified sampling (min_stories={min_stories}, sample_size={sample_size})...")

        # Count stories per subreddit
        subreddit_counts = stories_df['meta.subreddit'].value_counts()

        # Filter to subreddits with enough data
        eligible_subreddits = subreddit_counts[subreddit_counts >= min_stories].index.tolist()
        print(f"Found {len(eligible_subreddits)} subreddits with >= {min_stories} stories")

        # Sample stories from eligible subreddits
        np.random.seed(25)  # For reproducibility
        sampled_indices = []

        for subreddit in eligible_subreddits:
            subreddit_indices = stories_df[stories_df['meta.subreddit'] == subreddit].index
            sampled = np.random.choice(subreddit_indices, size=sample_size, replace=False)
            sampled_indices.extend(sampled)

        # Filter to sampled stories
        stories_df = stories_df.loc[sampled_indices].copy()
        print(f"Stratified dataset: {len(stories_df)} stories from {len(eligible_subreddits)} subreddits ({sample_size} per subreddit)")
    else:
        print("Using full dataset (no stratification)")

    print(f"Loaded {len(stories_df)} stories")
    print(f"Analyzing {len(dims)} dimensions: {dims}")

    # Get category counts for each dimension
    dim_category_counts = {}
    for dim in dims:
        unique_cats = get_unique_categories(stories_df, dim)
        dim_category_counts[dim] = len(unique_cats)
        print(f"  {dim}: {len(unique_cats)} categories")

    # Get unique subreddits
    subreddits = stories_df['meta.subreddit'].unique()
    print(f"\nFound {len(subreddits)} unique subreddits")

    # Compute entropy for each subreddit and dimension
    results = []

    for subreddit in subreddits:
        subreddit_entropies = {}
        subreddit_norm_entropies = {}

        for dim in dims:
            col_name = f'prompt_default${dim}_gen0_cats'

            if col_name not in stories_df.columns:
                print(f"Warning: Column {col_name} not found")
                continue

            # Get distribution and compute entropy
            dist = compute_subreddit_label_distribution(stories_df, subreddit, dim, col_name)
            raw_ent = compute_entropy_from_distribution(dist)
            norm_ent = compute_normalized_entropy(dist, dim_category_counts[dim])

            subreddit_entropies[f'{dim}_entropy'] = raw_ent
            subreddit_norm_entropies[f'{dim}_normalized_entropy'] = norm_ent

        # Compute average normalized entropy across all dimensions
        valid_norm_entropies = [v for v in subreddit_norm_entropies.values() if not np.isnan(v)]
        avg_normalized_entropy = np.mean(valid_norm_entropies) if valid_norm_entropies else np.nan

        # Compute group-level average normalized entropies
        group_entropies = {}
        for group_name, group_dims in DIMENSION_GROUPS.items():
            group_vals = [subreddit_norm_entropies.get(f'{dim}_normalized_entropy', np.nan)
                         for dim in group_dims if dim in dims]
            group_vals = [v for v in group_vals if not np.isnan(v)]
            group_entropies[f'{group_name}_avg_normalized_entropy'] = np.mean(group_vals) if group_vals else np.nan

        # Get story count for this subreddit
        story_count = len(stories_df[stories_df['meta.subreddit'] == subreddit])

        results.append({
            'subreddit': subreddit,
            'avg_normalized_entropy': avg_normalized_entropy,
            'story_count': story_count,
            **group_entropies,
            **subreddit_entropies,
            **subreddit_norm_entropies
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by average normalized entropy (descending)
    results_df = results_df.sort_values('avg_normalized_entropy', ascending=False)

    return results_df, dims, dim_category_counts

def plot_subreddit_entropy(results_df, output_path=f'{ENTROPY_RESULTS_DIR}/subreddit_normalized_entropy.pdf'):
    """Create bar plot of subreddit normalized entropy rankings."""
    # Set up the plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # Filter out NaN values
    plot_df = results_df[~results_df['avg_normalized_entropy'].isna()].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.3)))

    # Create horizontal bar plot
    bars = ax.barh(range(len(plot_df)), plot_df['avg_normalized_entropy'],
                   color='black', edgecolor='black')

    # Customize
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['subreddit'], fontsize=12)
    ax.set_xlabel('Average Normalized Entropy (Efficiency)', fontsize=14)
    ax.set_ylabel('Subreddit', fontsize=14)
    ax.set_title('Subreddit Ranking by Average Normalized Taxonomy Label Entropy', fontsize=16, pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--', color='gray')
    ax.set_xlim(0, 1.0)  # Normalized entropy is [0, 1]
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=12)

    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(row['avg_normalized_entropy'] + 0.01, i, f"{row['avg_normalized_entropy']:.3f}",
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")

    return plot_df

def plot_author_reader_scatterplot_generic(results_df, subreddit_dict, category_name='topic',
                                            output_path=None, use_multicolor=False):
    """Create scatterplot of author-centric vs reader-centric entropy colored by categories.

    Args:
        results_df: DataFrame with entropy results
        subreddit_dict: Dictionary mapping subreddits to categories (topics or community archetypes)
        category_name: Name of the category ('topic' or 'community_archetype')
        output_path: Path to save the plot
        use_multicolor: If True, use pie wedges for multiple categories
    """

    if output_path is None:
        output_path = f'{ENTROPY_RESULTS_DIR}/subreddit_entropy_{category_name}.pdf'

    # Filter out NaN values
    plot_df = results_df[~results_df['avg_normalized_entropy'].isna()].copy()
    plot_df = plot_df[~plot_df['author_centric_avg_normalized_entropy'].isna() &
                      ~plot_df['reader_centric_avg_normalized_entropy'].isna()]

    # Add category information
    plot_df['category'] = plot_df['subreddit'].map(subreddit_dict)

    # Extract data
    x = plot_df['reader_centric_avg_normalized_entropy'].values
    y = plot_df['author_centric_avg_normalized_entropy'].values

    # Compute line of best fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get all unique categories and assign colors
    all_categories = set()
    for cat in plot_df['category'].dropna():
        if isinstance(cat, list):
            all_categories.update(cat)
        elif isinstance(cat, str):
            cat_str = cat.strip("[]'\"")
            cats = [c.strip().strip("'\"") for c in cat_str.split(',')]
            all_categories.update(cats)

    all_categories = sorted(list(all_categories))
    palette = sns.color_palette('tab20', n_colors=len(all_categories))
    category_colors = dict(zip(all_categories, palette))

    # Track which categories are actually plotted for legend
    plotted_categories = set()

    # Plot each point with appropriate styling
    marker_size = 60
    radius = (marker_size / 100) ** 0.5 * 0.006
    has_unknown = False

    for idx, row in plot_df.iterrows():
        x_pos = row['reader_centric_avg_normalized_entropy']
        y_pos = row['author_centric_avg_normalized_entropy']
        cat = row['category']

        is_missing = (cat is None or
                     (isinstance(cat, float) and np.isnan(cat)) or
                     cat == '' or
                     (isinstance(cat, list) and len(cat) == 0))

        if is_missing:
            wedge = Wedge((x_pos, y_pos), radius, 0, 360,
                         facecolor='gray', edgecolor='black',
                         linewidth=0.5, alpha=0.6, zorder=2)
            ax.add_patch(wedge)
            has_unknown = True
        else:
            if isinstance(cat, list):
                cats = cat
            else:
                cat_str = str(cat).strip("[]'\"")
                cats = [c.strip().strip("'\"") for c in cat_str.split(',')]

            if len(cats) == 1:
                wedge = Wedge((x_pos, y_pos), radius, 0, 360,
                             facecolor=category_colors[cats[0]], edgecolor='black',
                             linewidth=0.5, alpha=0.6, zorder=2)
                ax.add_patch(wedge)
                plotted_categories.add(cats[0])
            else:
                if use_multicolor:
                    angles = np.linspace(0, 360, len(cats) + 1)
                    for i, c in enumerate(cats):
                        wedge = Wedge((x_pos, y_pos), radius, angles[i], angles[i+1],
                                     facecolor=category_colors[c], edgecolor='black',
                                     linewidth=0.5, alpha=0.6, zorder=2)
                        ax.add_patch(wedge)
                        plotted_categories.add(c)
                else:
                    wedge = Wedge((x_pos, y_pos), radius, 0, 360,
                                 facecolor=category_colors[cats[0]], edgecolor='black',
                                 linewidth=0.5, alpha=0.6, zorder=2)
                    ax.add_patch(wedge)
                    plotted_categories.add(cats[0])

    # Create legend, adding line breaks to long names
    def format_label(label):
        label = label.replace('_', ' ')
        words = label.split()
        if len(words) > 3:
            mid = len(words) // 2
            return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        elif len(label) > 20:
            return label.replace(' ', '\n', 1)
        return label

    legend_elements = [mpatches.Patch(facecolor=category_colors[cat],
                                     edgecolor='black', label=format_label(cat),
                                     alpha=0.6)
                      for cat in sorted(plotted_categories)]

    if has_unknown:
        legend_elements.append(mpatches.Patch(facecolor='gray',
                                             edgecolor='black', label='Unknown',
                                             alpha=0.6))

    # Set axis limits
    x_min, x_max = x.min() - 0.025, x.max() + 0.025
    y_min, y_max = y.min() - 0.025, y.max() + 0.025
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Annotate all points (bold labels)
    texts = []
    for idx, row in plot_df.iterrows():
        text = ax.text(row['reader_centric_avg_normalized_entropy'],
                      row['author_centric_avg_normalized_entropy'],
                      row['subreddit'],
                      fontsize=12, alpha=0.9)
        texts.append(text)

    adjust_text(texts,
                ax=ax,
                expand_points=(1.2, 1.2),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                force_points=0.3,
                force_text=0.5,
                lim=100)

    # Customize axes and title
    title = f'Subreddit Narrative Diversity'
    ax.set_xlabel('Reader-Centric Normalized Entropy', fontsize=18)
    ax.set_ylabel('Author-Centric Normalized Entropy', fontsize=18)
    ax.set_title(title, fontsize=22, pad=22)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2)
    ax.grid(alpha=0.3)

    # Restore axis limits after adjust_text
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1),
              loc='upper left', fontsize=12, frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved {category_name} scatterplot to {output_path}")
    print(f"  Correlation: r = {r_value:.3f}, R² = {r_value**2:.3f}, p = {p_value:.4f}")

def plot_entropy_bars_by_ca(results_df, subreddit_commArchs_dict,
                            output_path=f'{ENTROPY_RESULTS_DIR}/subreddit_entropy_by_ca.pdf'):
    """Create horizontal bar chart of subreddit entropy, sorted descending, with bars colored by community archetypes."""

    # Set up the plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # Prepare data
    plot_df = results_df[['subreddit', 'avg_normalized_entropy']].copy()
    plot_df = plot_df.sort_values('avg_normalized_entropy', ascending=True)  # Ascending for horizontal bars
    plot_df['comm_archs'] = plot_df['subreddit'].map(subreddit_commArchs_dict)

    # Get all unique community archetypes and assign colors
    all_archetypes = set()
    for comm_archs in plot_df['comm_archs'].dropna():
        if isinstance(comm_archs, list):
            all_archetypes.update(comm_archs)
        elif isinstance(comm_archs, str):
            comm_archs_str = comm_archs.strip("[]'\"")
            archs = [arch.strip().strip("'\"") for arch in comm_archs_str.split(',')]
            all_archetypes.update(archs)

    all_archetypes = sorted(list(all_archetypes))
    palette = sns.color_palette('tab20', n_colors=len(all_archetypes))
    archetype_colors = dict(zip(all_archetypes, palette))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 16))

    # Plot bars
    y_positions = range(len(plot_df))
    bar_height = 0.8

    for idx, (i, row) in enumerate(plot_df.iterrows()):
        entropy = row['avg_normalized_entropy']
        comm_archs = row['comm_archs']

        # Check if no archetype info
        is_missing = (comm_archs is None or
                     (isinstance(comm_archs, float) and np.isnan(comm_archs)) or
                     comm_archs == '' or
                     (isinstance(comm_archs, list) and len(comm_archs) == 0))

        if is_missing:
            # Gray bar for unknown
            ax.barh(y_positions[idx], entropy, bar_height, color='gray', edgecolor='black', linewidth=0.5)
        else:
            # Parse archetypes
            if isinstance(comm_archs, list):
                archs = comm_archs
            else:
                comm_archs_str = str(comm_archs).strip("[]'\"")
                archs = [arch.strip().strip("'\"") for arch in comm_archs_str.split(',')]

            if len(archs) == 1:
                # Single archetype - solid color bar
                ax.barh(y_positions[idx], entropy, bar_height,
                       color=archetype_colors[archs[0]], edgecolor='black', linewidth=0.5)
            else:
                # Multiple archetypes - create segmented bar
                segment_width = entropy / len(archs)
                for j, arch in enumerate(archs):
                    ax.barh(y_positions[idx], segment_width, bar_height,
                           left=j * segment_width,
                           color=archetype_colors[arch], edgecolor='black', linewidth=0.5)

    # Format labels
    def format_label(label):
        """Add line breaks to long archetype labels."""
        label = label.replace('_', ' ')
        words = label.split()
        if len(words) > 3:
            mid = len(words) // 2
            return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        elif len(label) > 20:
            return label.replace(' ', '\n', 1)
        return label

    # Create legend
    legend_elements = [mpatches.Patch(facecolor=archetype_colors[arch],
                                     edgecolor='black', label=format_label(arch))
                      for arch in all_archetypes]
    legend_elements.append(mpatches.Patch(facecolor='gray', edgecolor='black', label='Unknown'))

    # Customize
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df['subreddit'], fontsize=10)
    ax.set_xlabel('Average Normalized Entropy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Subreddit', fontsize=14, fontweight='bold')
    ax.set_title('Subreddit Entropy by Community Archetype', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved entropy bar chart to {output_path}")

def get_default_openai_generation_strategy(config):
    model_config = ModelConfig(model_name=config.models.openai_default)
    generation_config = GenerationConfig()
    return OpenaiGenerationStrategy(model_config=model_config,
                                    generation_config=generation_config)
if __name__ == "__main__":
    config = load_config(REPLICATION_CONFIG_PATH)

    stories_path = f"{config.dirs.data.corpus}/{SSF_DF_PATH}"

    # Initialize SubredditPreprocessor to get topic and community archetype info
    print("Loading subreddit metadata...")
    default_openai_generation_strategy = get_default_openai_generation_strategy(config)
    subreddit_preprocessor = SubredditPreprocessor(subreddits_path=SUBREDDITS_PATH,
                                                                        subreddit_desc_rules_path=SUBREDDIT_DESC_RULES_PATH,
                                                                        dir=config.dirs.data.subreddit_metadata,
                                                                        generation_strategy=default_openai_generation_strategy,
                                                                        force_rebuild=False)

    subreddit_topic_dict = subreddit_preprocessor.get_community_topic_dict()

    # Run analysis
    print("Computing subreddit normalized entropy analysis...")
    results_df, dims, dim_category_counts = analyze_subreddit_entropy(stories_path)

    # Save results
    results_path = f"{ENTROPY_RESULTS_DIR}/subreddit_normalized_entropy_analysis.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    # Display top results
    print("\nTop 10 subreddits by average normalized entropy:")
    print(results_df[['subreddit', 'avg_normalized_entropy', 'story_count']].head(10).to_string(index=False))

    print("\nBottom 10 subreddits by average normalized entropy:")
    print(results_df[['subreddit', 'avg_normalized_entropy', 'story_count']].tail(10).to_string(index=False))

    # Display group-level summary for top 5
    print("\nGroup-level breakdown for top 5 subreddits:")
    group_cols = ['subreddit'] + [f'{g}_avg_normalized_entropy' for g in DIMENSION_GROUPS.keys()]
    print(results_df[group_cols].head(5).to_string(index=False))

    # Create visualizations
    print("\nCreating visualizations...")
    plot_subreddit_entropy(results_df)

    # Create scatterplots for both topics and community archetypes
    print("\nGenerating scatterplots...")
    plot_author_reader_scatterplot_generic(results_df, subreddit_topic_dict,
                                          category_name='topic', use_multicolor=False)

    print("\nDone!")
