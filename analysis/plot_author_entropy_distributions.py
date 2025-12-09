import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from ast import literal_eval
from ssf.Constants import *
from ssf.Taxonomy import Taxonomy
from ssf.utils.TaxonomyDataUtils import get_cats_col_name
from ssf.Configs import load_config

def get_cats_converters(taxonomy):
    """Get converters for category list columns."""
    return {
        get_cats_col_name(PROMPT_COL_SUFFIX_FULL_CONTEXT, dim, "gen0"): literal_eval
        for dim in taxonomy.get_dims()
    }

def get_subreddit_label_distribution(stories_df, subreddit, dim):
    """Get label distribution for a subreddit and dimension."""
    col_name = f'prompt_default${dim}_gen0_cats'
    subreddit_stories = stories_df[stories_df['meta.subreddit'] == subreddit]

    all_labels = []
    for _, row in subreddit_stories.iterrows():
        labels = row[col_name]
        # Lists are already parsed via converters
        if isinstance(labels, list):
            all_labels.extend(labels)

    if not all_labels:
        return {}

    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    label_probs = {label: count / total for label, count in label_counts.items()}

    return label_probs

def plot_author_entropy_comparison(stories_path, output_path=f'{ENTROPY_RESULTS_DIR}/author_entropy_distributions.pdf'):
    """
    Plot distributions for the three author-centric dimensions comparing
    single highest and single lowest entropy subreddit with overlaid bars.
    """
    # Load taxonomy to get converters
    taxonomy = Taxonomy(taxonomy_dir=TAXONOMY_DIR)

    # Load data with converters
    stories_df = pd.read_csv(stories_path, converters=get_cats_converters(taxonomy))
    results_df = pd.read_csv(f'{ENTROPY_RESULTS_DIR}/subreddit_normalized_entropy_analysis.csv')

    # Get single highest and single lowest author-centric entropy subreddit
    results_df_sorted = results_df.sort_values('author_centric_avg_normalized_entropy', ascending=False)

    high_subreddit = results_df_sorted['subreddit'].iloc[0]
    low_subreddit = results_df_sorted['subreddit'].iloc[-1]

    high_entropy_val = results_df_sorted['author_centric_avg_normalized_entropy'].iloc[0]
    low_entropy_val = results_df_sorted['author_centric_avg_normalized_entropy'].iloc[-1]

    high_story_count = results_df_sorted.loc[results_df_sorted['subreddit'] == high_subreddit, 'story_count'].iloc[0]
    low_story_count = results_df_sorted.loc[results_df_sorted['subreddit'] == low_subreddit, 'story_count'].iloc[0]

    print(f"Highest author entropy: {high_subreddit} ({high_entropy_val:.3f}, n={high_story_count})")
    print(f"Lowest author entropy: {low_subreddit} ({low_entropy_val:.3f}, n={low_story_count})")

    # Author-centric dimensions
    author_dims = ['overall_goal', 'narrative_intent', 'author_emotional_response']

    # Create figure with 1 row, 3 columns (vertical bar plots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for dim_idx, dim in enumerate(author_dims):
        ax = axes[dim_idx]

        # Get distributions for both subreddits
        dist_high = get_subreddit_label_distribution(stories_df, high_subreddit, dim)
        dist_low = get_subreddit_label_distribution(stories_df, low_subreddit, dim)

        # Get union of all categories
        all_categories = set(dist_high.keys()) | set(dist_low.keys())

        # Create aligned dictionaries
        probs_high = {cat: dist_high.get(cat, 0) for cat in all_categories}
        probs_low = {cat: dist_low.get(cat, 0) for cat in all_categories}

        # Sort by low entropy probabilities (descending)
        sorted_cats = sorted(all_categories, key=lambda x: probs_low[x], reverse=True)

        # Prepare data for plotting
        x_pos = np.arange(len(sorted_cats))
        high_vals = [probs_high[cat] for cat in sorted_cats]
        low_vals = [probs_low[cat] for cat in sorted_cats]

        # Swap the visual styles:
        # → High entropy = hatched (striped)
        # → Low entropy = solid gray
        bar_width = 0.35
        ax.bar(x_pos - bar_width/2, high_vals, width=bar_width,
               color='white', edgecolor='black', linewidth=1.2, hatch='///',
               label=f'{high_subreddit} (high entropy: {high_entropy_val:.3f})')
        ax.bar(x_pos + bar_width/2, low_vals, width=bar_width,
               color='gray', edgecolor='black', linewidth=1.2,
               label=f'{low_subreddit} (low entropy: {low_entropy_val:.3f})')

        # Customize
        ax.set_xticks(x_pos)
        ax.set_xticklabels([cat.replace('_', ' ') for cat in sorted_cats],
                           rotation=45, ha='right', fontsize=13)
        ax.set_ylabel('Probability', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=13, width=1.2)
        ax.set_title(f"{dim.replace('_', ' ').title()}",
                     fontsize=15, fontweight='bold', pad=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)

        # Set consistent y-axis limit
        max_prob = max(max(high_vals), max(low_vals))
        ax.set_ylim(0, max_prob * 1.15)

    plt.suptitle('Author-Centric Dimension Distributions: Highest vs Lowest Entropy Subreddits',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved distribution comparison to {output_path}")

if __name__ == "__main__":
    config = load_config(REPLICATION_CONFIG_PATH)
    stories_path = f"{config.dirs.data.corpus}/{SSF_DF_PATH}"

    print("Creating author entropy distribution comparison...")
    plot_author_entropy_comparison(stories_path)
    print("Done!")
