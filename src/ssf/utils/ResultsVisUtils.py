import pandas as pd
import os
from ssf.RatingOptions import RatingOptions
import matplotlib.pyplot as plt
import seaborn as sns
from ssf.Constants import *
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from typing import Dict, List
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from ssf.utils import MetricUtils

def vis_ctx_sum_ann(stats_dict, output_path):
    """
    Create black and white bar chart for annotation statistics (annotator 1 only).

    Args:
        stats_dict: Dictionary with structure {row_name: {consistency_mean, consistency_std, relevance_mean, relevance_std}}
        output_path: Path to save the plot
    """
    # Prepare data
    categories = ["conversation", "initial post", "subreddit purpose", "subreddit rules/norms/values"]

    data = []
    for category in categories:
        data.append({
            'Summary Type': category,
            'Metric': 'Consistency',
            'Score': stats_dict[category]['consistency_mean'],
            'Error': stats_dict[category]['consistency_std']
        })
        data.append({
            'Summary Type': category,
            'Metric': 'Relevance',
            'Score': stats_dict[category]['relevance_mean'],
            'Error': stats_dict[category]['relevance_std']
        })

    df = pd.DataFrame(data)

    # Set up black and white style
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped positions for bars
    n_categories = len(categories)
    bar_width = 0.35
    group_gap = 0.2

    x = []
    for i in range(n_categories):
        x.append(i * (bar_width * 2 + group_gap))

    # Get data for each metric
    cons_scores = [stats_dict[cat]['consistency_mean'] for cat in categories]
    cons_errors = [stats_dict[cat]['consistency_std'] for cat in categories]
    rel_scores = [stats_dict[cat]['relevance_mean'] for cat in categories]
    rel_errors = [stats_dict[cat]['relevance_std'] for cat in categories]

    # Create positions
    cons_positions = x
    rel_positions = [xi + bar_width for xi in x]

    # Create bars
    ax.bar(cons_positions, cons_scores, bar_width,
           color='black', yerr=cons_errors,
           capsize=3, error_kw={'linewidth': 1})

    ax.bar(rel_positions, rel_scores, bar_width,
           color='gray', yerr=rel_errors,
           capsize=3, error_kw={'linewidth': 1})

    # Add text labels on the bars
    for i, (cons_pos, rel_pos) in enumerate(zip(cons_positions, rel_positions)):
        ax.text(cons_pos, -0.3, 'Consistency', ha='center', va='top', fontsize=10, rotation=0)
        ax.text(rel_pos, -0.3, 'Relevance', ha='center', va='top', fontsize=10, rotation=0)

    # Customize the plot
    ax.set_ylabel('Rating Score', fontsize=18)
    ax.set_title('Context Summarization Annotation Results (Annotator 1)', fontsize=20, pad=20)
    ax.set_xticks([xi + bar_width/2 for xi in x])
    ax.set_xticklabels(categories, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Add horizontal line at y=5 for max rating
    ax.axhline(y=5, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Set y-axis limits with extra space at bottom for labels
    ax.set_ylim(-0.6, 5.5)

    # Remove spines
    sns.despine()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Bar chart saved to {output_path}")

def gpt4o_x_prolific_hw(autoResultsDict, out_path):
    dim_aggMetricsDict_dict = autoResultsDict['dim_aggMetricsDict_dict']
    metricName_overallScore_dict = autoResultsDict['metricName_overallScore_dict']
    
    metrics = ['cosine_similarity', 'bert_score', 'bleu', 'meteor', 'n']
    
    # Start building LaTeX table
    table_lines = []
    table_lines.append("\\begin{table*}[h]")
    table_lines.append("\\small")
    table_lines.append("\\centering")
    
    # Create column specification (l for dimension names, c for each metric)
    col_spec = "l" + "c" * len(metrics)
    table_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    table_lines.append("\\toprule")
    
    # Header row
    header = "Dimension & " + " & ".join([
        "Cosine Sim.", "BERTScore", "BLEU", "METEOR", "N"
    ]) + " \\\\"
    table_lines.append(header)
    table_lines.append("\\midrule")
    
    # Data rows for each dimension
    for dim, aggMetricsDict in dim_aggMetricsDict_dict.items():
        row_data = [dim.replace("_", "\\_")]  # Escape underscores for LaTeX
        for metric in metrics:
            if metric in aggMetricsDict:
                value = aggMetricsDict[metric]
                if metric == 'n':
                    row_data.append(f"{int(value)}")
                else:
                    row_data.append(f"{value:.3f}")
            else:
                row_data.append("--")
        
        row = " & ".join(row_data) + " \\\\"
        table_lines.append(row)
    
    table_lines.append("\\midrule")
    
    # Overall results row
    overall_row_data = ["Overall"]
    for metric in metrics:
        if metric == 'n':
            # Use count from overall_metricName_counts_dict (should be same for all metrics)
            value = metricName_overallScore_dict['cosine_similarity']
            overall_row_data.append(f"{int(value)}")
        elif metric in metricName_overallScore_dict:
            value = metricName_overallScore_dict[metric]
            overall_row_data.append(f"{value:.3f}")
        else:
            overall_row_data.append("--")
    
    overall_row = " & ".join(overall_row_data) + " \\\\"
    table_lines.append(overall_row)
    
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\caption{Comparison of human-written inferences to GPT-4o generated inferences. We report cosine similarity, BERTScore \\cite{Zhang2019-tb}, $BLEU_3$ \\cite{Papineni2002-hn}, and METEOR \\cite{Banerjee2005-dg}.}")
    table_lines.append("\\label{tab:gpt4o_x_prolific_hw}")
    table_lines.append("\\end{table*}")


    latex_table = "\n".join(table_lines)

    # Write to file
    with open(out_path, 'w') as f:
        f.write(latex_table)

    return latex_table

def get_cluster_order(matrix, method='average', metric='euclidean'):
    """
    Get the clustering order without displaying the plot.
    Returns the row and column orders from hierarchical clustering.
    """
    mat = matrix.fillna(0).astype(float)

    # Compute linkage for rows
    row_linkage = linkage(pdist(mat, metric=metric), method=method)

    # Compute linkage for columns (if different from rows)
    if mat.shape[0] == mat.shape[1] and (mat.index == mat.columns).all():
        # Square matrix with same row/col labels - use same clustering
        col_linkage = row_linkage
    else:
        # Different dimensions - cluster columns separately
        col_linkage = linkage(pdist(mat.T, metric=metric), method=method)

    # Get the order from linkage using scipy's leaves_list
    from scipy.cluster.hierarchy import leaves_list
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    return row_order, col_order

def plot_ctx_sum_ann_bars(ann1_stats_dict, ann2_stats_dict, output_path, ann1_n_samples=50, ann2_n_samples=30):
    """
    Create black and white bar chart with side-by-side bars for two annotators.
    Ann1 uses solid bars, Ann2 uses hatched bars.

    Args:
        ann1_stats_dict: Dictionary for annotator 1 with structure {row_name: {consistency_mean, consistency_std, relevance_mean, relevance_std}}
        ann2_stats_dict: Dictionary for annotator 2 with same structure
        output_path: Path to save the plot
        ann1_n_samples: Number of samples for annotator 1 (default 50)
        ann2_n_samples: Number of samples for annotator 2 (default 30)
    """
    # Prepare data in long format
    data = []

    # Categories we want to plot (only those that have ann2 data)
    categories_to_plot = ["subreddit purpose", "subreddit values/norms", "initial post", "conversation history"]

    for category in categories_to_plot:
        # Annotator 1 - Consistency
        data.append({
            'Summary Type': category,
            'Metric': 'Consistency',
            'Annotator': 'Ann1',
            'Score': ann1_stats_dict[category]['consistency_mean'],
            'Error': ann1_stats_dict[category]['consistency_std']
        })
        # Annotator 1 - Relevance
        data.append({
            'Summary Type': category,
            'Metric': 'Relevance',
            'Annotator': 'Ann1',
            'Score': ann1_stats_dict[category]['relevance_mean'],
            'Error': ann1_stats_dict[category]['relevance_std']
        })
        # Annotator 2 - Consistency
        data.append({
            'Summary Type': category,
            'Metric': 'Consistency',
            'Annotator': 'Ann2',
            'Score': ann2_stats_dict[category]['consistency_mean'],
            'Error': ann2_stats_dict[category]['consistency_std']
        })
        # Annotator 2 - Relevance
        data.append({
            'Summary Type': category,
            'Metric': 'Relevance',
            'Annotator': 'Ann2',
            'Score': ann2_stats_dict[category]['relevance_mean'],
            'Error': ann2_stats_dict[category]['relevance_std']
        })

    df = pd.DataFrame(data)

    # Set up the plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped positions for bars
    n_categories = len(categories_to_plot)
    n_metrics = 2  # Consistency and Relevance
    n_annotators = 2

    bar_width = 0.15
    group_width = bar_width * n_annotators * n_metrics
    group_gap = 0.3

    x = []
    for i in range(n_categories):
        x.append(i * (group_width + group_gap))

    # Plot bars manually for better control
    positions = {
        'Consistency_Ann1': [],
        'Consistency_Ann2': [],
        'Relevance_Ann1': [],
        'Relevance_Ann2': []
    }

    for i, category in enumerate(categories_to_plot):
        base_x = x[i]
        positions['Consistency_Ann1'].append(base_x)
        positions['Consistency_Ann2'].append(base_x + bar_width)
        positions['Relevance_Ann1'].append(base_x + bar_width * 2.5)
        positions['Relevance_Ann2'].append(base_x + bar_width * 3.5)

    # Get data for each group
    def get_values(category, metric, annotator):
        mask = (df['Summary Type'] == category) & (df['Metric'] == metric) & (df['Annotator'] == annotator)
        return df[mask]['Score'].values[0], df[mask]['Error'].values[0]

    # Plot Consistency bars
    cons_ann1_scores = [get_values(cat, 'Consistency', 'Ann1')[0] for cat in categories_to_plot]
    cons_ann1_errors = [get_values(cat, 'Consistency', 'Ann1')[1] for cat in categories_to_plot]
    cons_ann2_scores = [get_values(cat, 'Consistency', 'Ann2')[0] for cat in categories_to_plot]
    cons_ann2_errors = [get_values(cat, 'Consistency', 'Ann2')[1] for cat in categories_to_plot]

    # Plot Relevance bars
    rel_ann1_scores = [get_values(cat, 'Relevance', 'Ann1')[0] for cat in categories_to_plot]
    rel_ann1_errors = [get_values(cat, 'Relevance', 'Ann1')[1] for cat in categories_to_plot]
    rel_ann2_scores = [get_values(cat, 'Relevance', 'Ann2')[0] for cat in categories_to_plot]
    rel_ann2_errors = [get_values(cat, 'Relevance', 'Ann2')[1] for cat in categories_to_plot]

    # Create bars - solid for Ann1, hatched for Ann2
    ax.bar(positions['Consistency_Ann1'], cons_ann1_scores, bar_width,
           color='black', label='Consistency Ann1', yerr=cons_ann1_errors,
           capsize=3, error_kw={'linewidth': 1}, edgecolor='black')

    ax.bar(positions['Consistency_Ann2'], cons_ann2_scores, bar_width,
           color='white', edgecolor='black', hatch='///', label='Consistency Ann2',
           yerr=cons_ann2_errors, capsize=3, error_kw={'linewidth': 1}, linewidth=1.5)

    ax.bar(positions['Relevance_Ann1'], rel_ann1_scores, bar_width,
           color='gray', label='Relevance Ann1', yerr=rel_ann1_errors,
           capsize=3, error_kw={'linewidth': 1}, edgecolor='gray')

    ax.bar(positions['Relevance_Ann2'], rel_ann2_scores, bar_width,
           color='white', edgecolor='gray', hatch='///', label='Relevance Ann2',
           yerr=rel_ann2_errors, capsize=3, error_kw={'linewidth': 1}, linewidth=1.5)

    # Customize the plot
    ax.set_ylabel('Rating (Mean)', fontsize=14)
    ax.set_title('Context Summary Ratings by Type', fontsize=16, pad=20)

    # Center labels in the middle of all 4 bars for each category
    # Calculate the center position for each group of 4 bars
    center_positions = []
    for i in range(n_categories):
        # Get the leftmost and rightmost bar positions for this category
        leftmost = positions['Consistency_Ann1'][i]
        rightmost = positions['Relevance_Ann2'][i]
        center = (leftmost + rightmost) / 2
        center_positions.append(center)

    ax.set_xticks(center_positions)
    ax.set_xticklabels(categories_to_plot, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Add secondary x-axis label
    ax.set_xlabel('Summary Type', fontsize=14, labelpad=10)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Place legend outside the plot area to the right
    ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='black')

    # Add horizontal line at y=5 for max rating
    ax.axhline(y=5, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Set y-axis limits (no extra space needed since we removed labels)
    ax.set_ylim(0, 5.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Bar chart saved to {output_path}")

def plot_npmi_heatmap(matrix, key, clustered=False, output_dir=RESULTS_DIR):
    """Plot and save NPMI heatmap with optional clustering."""
    # Extract dimension names for labels
    dim_x, dim_y = key.split('$')

    # Apply clustering if requested
    if clustered:
        row_order, col_order = get_cluster_order(matrix)
        plot_matrix = matrix.iloc[row_order, col_order]
        title_suffix = " (clustered)"
        filename_suffix = "_clustered"
    else:
        plot_matrix = matrix
        title_suffix = ""
        filename_suffix = ""

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create heatmap
    sns.heatmap(
        plot_matrix,
        ax=ax,
        cmap="viridis",
        annot=True,
        square=True,
        cbar_kws={'shrink': 0.8},
        fmt='.2f'
    )

    # Set title and axis labels
    ax.set_title(f"Inter-dimension Associations: {dim_x} × {dim_y}{title_suffix}", fontsize=12, pad=20)
    ax.set_xlabel(f"{dim_y.replace('_', ' ').title()}", fontsize=11)
    ax.set_ylabel(f"{dim_x.replace('_', ' ').title()}", fontsize=11)

    # Formatting
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), va='center')

    # Save figure
    filename = key.replace('$', '_')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/npmi/{filename}_npmi_heatmap{filename_suffix}.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()

    # print(f"Saved heatmap for {key} as {filename}_npmi_heatmap{filename_suffix}.pdf")

def _calculate_overall_proportions(dim_preds):
    """Calculate overall proportions across all dimensions."""
    all_scores = [score for scores in dim_preds.values() for score in scores]
    
    if not all_scores:
        return {i: 0 for i in range(1, 5)}
    
    score_counts = {i: all_scores.count(i) for i in range(1, 5)}
    overall_proportions = {i: count / len(all_scores) for i, count in score_counts.items()}
    
    return overall_proportions

def plot_option_1_hatching(results_dict: Dict, dimensions: List[str], out_path: str):
    """Option 1: Use hatching patterns to differentiate datasets."""
    # Set up the plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    # Map model names
    model_name_mapping = {'Model 1': 'GPT-4o', 'Model 2': 'SSF-Generator'}
    dataset_names = [model_name_mapping.get(name, name) for name in results_dict.keys()]
    
    rating_options = RatingOptions.all()
    colors = ["#e02117ff", "#e18719ff", "#487FC6ff", "#4a5d9cff"]  # All colors at full opacity
    patterns = ['', '///']  # Solid and diagonal lines
    
    extended_dimensions = ['overall'] + dimensions
    fig, ax = plt.subplots(figsize=(18, 6))
    
    x = np.arange(len(extended_dimensions))
    width = 0.42  # Slightly wider bars
    spacing = 0.05  # Narrower spacing between groups
    
    # Create legend handles for colors (rating scores)
    color_legend_handles = []
    
    for dataset_idx, (original_name, dataset_name) in enumerate(zip(results_dict.keys(), dataset_names)):
        dim_proportions = results_dict[original_name]['dim_proportions']
        dim_preds = results_dict[original_name]['dim_preds']
        overall_proportions = _calculate_overall_proportions(dim_preds)
        
        positions = x * (1 + spacing) + (dataset_idx - 0.5) * width
        bottom = np.zeros(len(extended_dimensions))
        
        for score_idx in range(1, 5):  # Keep original bar stacking order
            proportions = []
            proportions.append(overall_proportions[score_idx] * 100)
            for dim in dimensions:
                proportions.append(dim_proportions[dim][score_idx] * 100)
            
            bars = ax.bar(positions, proportions, width, 
                         bottom=bottom, 
                         color=colors[score_idx-1],
                         hatch=patterns[dataset_idx],
                         edgecolor='white', linewidth=0.5)
            
            # Create color legend handles only once (from first dataset)
            if dataset_idx == 0:
                # Use strategic newlines for long rating option names
                rating_label = rating_options[score_idx-1]
                if 'or' in rating_label:
                    rating_label = rating_label.replace(' or ', ' or\n')
                elif 'and' in rating_label:
                    rating_label = rating_label.replace(' and ', ' and\n')
                elif len(rating_label) > 12:  # For other long labels
                    words = rating_label.split()
                    if len(words) >= 2:
                        mid = len(words) // 2
                        rating_label = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                
                color_legend_handles.append(Rectangle((0,0),1,1, 
                                                    facecolor=colors[score_idx-1],
                                                    label=rating_label))
            
            bottom += proportions
        
        # Add only n= labels (no model names)
        for i, dim in enumerate(extended_dimensions):
            n_value = (sum(len(scores) for scores in dim_preds.values()) 
                      if dim == 'overall' else len(dim_preds[dim]))
            ax.text(positions[i], 104, f'n={n_value}', 
                   ha='center', va='bottom', fontsize=9)
    
    # Add color-based legend for rating scores - closer to plot
    # Reverse the order of legend handles to match the visual stacking
    color_legend_handles.reverse()
    ax.legend(handles=color_legend_handles, title='Rating Scores', 
             bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, title_fontsize=14)
    
    # Add model pattern legend - positioned right underneath the rating scores legend
    pattern_legend_handles = [Rectangle((0,0),1,1, facecolor='gray', hatch=patterns[i],
                                       label=dataset_names[i]) 
                             for i in range(len(dataset_names))]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=pattern_legend_handles, title='Models', 
              bbox_to_anchor=(1.02, 0.45), loc='upper left', fontsize=12, title_fontsize=14)
    
    _finalize_plot(ax, extended_dimensions, spacing, out_path, 
                  'Inference Plausibility Ratings by Dimension and Model')

def _finalize_plot(ax, extended_dimensions, spacing, out_path, title):
    """Helper function to finalize plot styling."""
    ax.set_xlabel('Dimensions', fontsize=16)
    ax.set_ylabel('Percentage of Ratings (%)', fontsize=16)
    ax.set_title(title, fontsize=18, pad=20)
    
    ax.set_xticks(np.arange(len(extended_dimensions)) * (1 + spacing))
    
    # Format x-axis labels with newlines instead of underscores, no rotation
    formatted_labels = [dim.replace('_', '\n') for dim in extended_dimensions]
    
    # Make the first label (overall) bold
    formatted_labels[0] = r'$\mathbf{' + formatted_labels[0] + '}$'
    
    ax.set_xticklabels(formatted_labels, ha='center', fontsize=12)
    ax.set_ylim(0, 115)
    
    # Add this line to set y-axis ticks every 10%
    # ax.set_yticks(np.arange(0, 120, 10))
    ax.set_yticks(np.arange(0, 110, 10))
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_similarity_correlation(df1, df2, x_label, y_label, title_prefix=None, figsize=(8, 6), 
                               annotate_top_k=5, annotate_agreement=3, save_path=None):
    """
    Create rank-based correlation plot between two similarity matrices with annotations.
    
    Args:
        df1: First similarity DataFrame (square matrix)
        df2: Second similarity DataFrame (square matrix) 
        x_label: Label for x-axis
        y_label: Label for y-axis
        title_prefix: Optional prefix for title (defaults to f"{x_label} Rank x {y_label} Rank")
        figsize: Figure size tuple
        annotate_top_k: Number of top positive and negative rank changes to annotate (0 to disable)
        annotate_agreement: Number of agreement cases to annotate (0 to disable)
        save_path: Optional path to save the plot (e.g., 'results/subreddit_similarity.pdf')
    """
    # Set up the plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    # Extract upper triangular values (excluding diagonal)
    triu_indices = np.triu_indices_from(df1, k=1)
    sim1_flat = df1.values[triu_indices]
    sim2_flat = df2.values[triu_indices]
    
    # Compute Spearman correlation on original data
    spearman_r, spearman_p = spearmanr(sim1_flat, sim2_flat)
    
    # Convert to ranks for plotting (higher similarity = lower rank number)
    sim1_ranks = len(sim1_flat) - sim1_flat.argsort().argsort()
    sim2_ranks = len(sim2_flat) - sim2_flat.argsort().argsort()
    
    # Get subreddit names for pair identification
    subreddit_names = df1.index.tolist()
    pairs = [(subreddit_names[i], subreddit_names[j]) for i, j in zip(*triu_indices)]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Beautiful scatter plot
    plt.scatter(sim1_ranks, sim2_ranks, 
               alpha=0.7, s=50, color='steelblue', edgecolor='white', linewidth=0.5)
    
    # Add diagonal line (perfect rank agreement)
    min_rank = min(sim1_ranks.min(), sim2_ranks.min())
    max_rank = max(sim1_ranks.max(), sim2_ranks.max())
    plt.plot([min_rank, max_rank], [min_rank, max_rank], 
             color='crimson', linestyle='--', alpha=0.8, linewidth=2, label='Perfect Rank Agreement')
    
    # Add annotations for top rank changes if requested
    if annotate_top_k > 0:
        top_positive, top_negative = MetricUtils.get_top_positive_negative_changes(df1, df2, x_label, y_label, annotate_top_k)
        
        # Combine positive and negative changes
        all_changes = pd.concat([top_positive, top_negative])
        
        # For each change, find its index in the flattened arrays and get plotting coordinates
        for _, row in all_changes.iterrows():
            pair = row['pair']
            sub1, sub2 = pair
            
            # Find the index of this pair in the original arrays
            pair_idx = pairs.index(pair)
            
            # Get the plotting coordinates (these are the actual rank positions)
            plot_x = sim1_ranks[pair_idx]
            plot_y = sim2_ranks[pair_idx]
            
            # Choose color based on sign of rank gap
            color = 'lightgreen' if row['rank_gap_signed'] > 0 else 'lightcoral'
            
            # Add annotation at the correct position
            plt.annotate(f"{sub1}–{sub2}", 
                        xy=(plot_x, plot_y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    # Add annotations for agreement cases if requested
    if annotate_agreement > 0:
        high_agreement, low_agreement = MetricUtils.get_agreement_cases(df1, df2, x_label, y_label, annotate_agreement)
        
        # High similarity agreement annotations (spread them out)
        for i, (_, row) in enumerate(high_agreement.iterrows()):
            pair = row['pair']
            sub1, sub2 = pair
            
            # Find the index of this pair in the original arrays
            pair_idx = pairs.index(pair)
            
            # Get the plotting coordinates
            plot_x = sim1_ranks[pair_idx]
            plot_y = sim2_ranks[pair_idx]
            
            # Spread out the annotations with varying offsets
            offset_x = 15 + (i * 25)  # Increasing x offset
            offset_y = 10 + (i * 15)  # Increasing y offset
            
            # Add annotation with larger offset
            plt.annotate(f"{sub1}–{sub2}", 
                        xy=(plot_x, plot_y),
                        xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=8, fontweight='bold', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='orange', alpha=0.6))
        
        # Low similarity agreement annotations (spread them out)
        for i, (_, row) in enumerate(low_agreement.iterrows()):
            pair = row['pair']
            sub1, sub2 = pair
            
            # Find the index of this pair in the original arrays
            pair_idx = pairs.index(pair)
            
            # Get the plotting coordinates
            plot_x = sim1_ranks[pair_idx]
            plot_y = sim2_ranks[pair_idx]
            
            # Spread out the annotations with varying offsets (in opposite direction)
            offset_x = -15 - (i * 25)  # Decreasing x offset
            offset_y = -10 - (i * 15)  # Decreasing y offset
            
            # Add annotation with larger offset
            plt.annotate(f"{sub1}–{sub2}", 
                        xy=(plot_x, plot_y),
                        xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=8, fontweight='bold', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
    
    # Flip axes so most similar (rank 1) is at top-right
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    
    # Remove tick marks and numbers
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    
    # Descriptive axis labels emphasizing relative nature
    plt.xlabel(f"Relative {x_label} →", fontsize=14)
    plt.ylabel(f"Relative {y_label} →", fontsize=14)
    
    # Set title
    if title_prefix is None:
        title_prefix = f"{y_label} Rank by {x_label} Rank"
    
    plt.title(title_prefix, fontsize=16, pad=20)
    
    # Add legend for diagonal line with updated style
    plt.legend(frameon=True, edgecolor='black', fontsize=12)
    
    # Grid styling
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print Spearman correlation info
    print(f"Correlation Analysis: {x_label} vs {y_label}")
    print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.2g}")
    
    # Print top rank changes if annotations were added
    if annotate_top_k > 0:
        top_positive, top_negative = MetricUtils.get_top_positive_negative_changes(df1, df2, x_label, y_label, annotate_top_k)
        
        print(f"\nTop {annotate_top_k} pairs where {x_label} ranks higher than {y_label}:")
        for i, (_, row) in enumerate(top_positive.iterrows(), 1):
            sub1, sub2 = row['pair']
            rank_gap = row['rank_gap_signed']
            print(f"{i}. {sub1} <-> {sub2}: Rank gap = +{rank_gap:.0f}")
        
        print(f"\nTop {annotate_top_k} pairs where {y_label} ranks higher than {x_label}:")
        for i, (_, row) in enumerate(top_negative.iterrows(), 1):
            sub1, sub2 = row['pair']
            rank_gap = row['rank_gap_signed']
            print(f"{i}. {sub1} <-> {sub2}: Rank gap = {rank_gap:.0f}")
    
    # Print agreement cases if annotations were added
    if annotate_agreement > 0:
        high_agreement, low_agreement = MetricUtils.get_agreement_cases(df1, df2, x_label, y_label, annotate_agreement)
        
        print(f"\nTop {annotate_agreement} pairs with strong agreement on HIGH similarity:")
        for i, (_, row) in enumerate(high_agreement.iterrows(), 1):
            sub1, sub2 = row['pair']
            rank_gap = row['rank_gap_signed']
            print(f"{i}. {sub1} <-> {sub2}: Rank gap = {rank_gap:+.0f}")
        
        print(f"\nTop {annotate_agreement} pairs with strong agreement on LOW similarity:")
        for i, (_, row) in enumerate(low_agreement.iterrows(), 1):
            sub1, sub2 = row['pair']
            rank_gap = row['rank_gap_signed']
            print(f"{i}. {sub1} <-> {sub2}: Rank gap = {rank_gap:+.0f}")
    
    print()

def plot_side_by_side_counts_per_dimension(df, subdims, filter_col, save_individual=False, output_dir=COUNTS_RESULTS_DIR):
    """
    Plot side-by-side bar charts comparing proportions between full dataset vs stratified dataset for each dimension.
    Bars are sorted by magnitude in descending order.

    Args:
        df: DataFrame to analyze (must have {dim}_cats columns)
        subdims: Dictionary mapping dimension names to their subcategories
        filter_col: Column name to filter by
        save_individual: If True, save individual plots by dimension name
        output_dir: Directory to save plots (default: COUNTS_RESULTS_DIR from Constants)

    Example:
        >>> subdims = {dim: list(taxonomy.dim_data_dict[dim].keys()) for dim in taxonomy.get_dims()}
        >>> plot_side_by_side_counts_per_dimension(
        ...     df=tc_analysis_df,
        ...     subdims=subdims,
        ...     filter_col='is_sampled_for_subreddit',
        ...     save_individual=True
        ... )
    """
    from ssf.utils.TaxonomyDataUtils import get_label_proportions

    # Set up the plot style
    plt.rcdefaults()  # Reset to defaults first
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    num_dims = len(subdims)
    rows = 4  # Four rows
    cols = 3  # Three columns

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, (dim, cats) in enumerate(subdims.items()):
        # Get proportions for full dataset and stratified dataset
        # Note: df has columns named {dim}_cats
        col_name = f"{dim}_cats"
        full_props = get_label_proportions(df, col_name)
        stratified_props = get_label_proportions(df, col_name, filter_col, True)

        # Get all unique labels from both datasets
        all_labels = sorted(set(full_props.index) | set(stratified_props.index))

        # Align both series to have the same labels
        full_props = full_props.reindex(all_labels, fill_value=0)
        stratified_props = stratified_props.reindex(all_labels, fill_value=0)

        # Sort by the sum of proportions (descending)
        combined_props = full_props + stratified_props
        sorted_indices = combined_props.sort_values(ascending=False).index

        # Reorder both series
        full_props = full_props.reindex(sorted_indices)
        stratified_props = stratified_props.reindex(sorted_indices)

        # Create side-by-side bar plot
        x = np.arange(len(sorted_indices))
        width = 0.35

        bars1 = axes[i].bar(x - width/2, full_props.values, width,
                           label='SSF-Corpus', color='black', edgecolor='black')
        bars2 = axes[i].bar(x + width/2, stratified_props.values, width,
                           label='SSF-Corpus-Stratified', color='white', edgecolor='black', hatch='///', linewidth=1.5)

        axes[i].set_title(f"{dim}", fontsize=18)
        axes[i].set_ylabel("Proportion", fontsize=16)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(sorted_indices, rotation=45, ha='right', va='top')
        axes[i].tick_params(axis='y', labelsize=14)
        axes[i].tick_params(axis='x', labelsize=14, length=8, width=2)
        axes[i].legend(fontsize=13)

        # Save individual plot
        if save_individual:
            individual_fig, individual_ax = plt.subplots(figsize=(10, 6))
            individual_ax.bar(x - width/2, full_props.values, width,
                            label='SSF-Corpus', color='black', edgecolor='black')
            individual_ax.bar(x + width/2, stratified_props.values, width,
                            label='SSF-Corpus-Stratified', color='white', edgecolor='black', hatch='///', linewidth=1.5)

            individual_ax.set_title(f"{dim}: Subdimension Frequencies", fontsize=18)
            individual_ax.set_ylabel("Proportion", fontsize=16)
            individual_ax.set_xticks(x)
            individual_ax.set_xticklabels(sorted_indices, rotation=45, ha='right', va='top', fontsize=13)
            individual_ax.legend(fontsize=12, title_fontsize=14)

            individual_fig.tight_layout()
            individual_fig.savefig(f"{output_dir}/{dim}_counts.pdf", bbox_inches='tight', dpi=300)
            plt.close(individual_fig)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.savefig(f"{output_dir}/dimension_counts.pdf", bbox_inches='tight')
    plt.show()


def plot_counts_by_community_comparison(df, filter_col):
    """
    Plot side-by-side bar charts comparing community counts for full vs stratified datasets.
    Communities are sorted by magnitude in descending order.

    Args:
        df: DataFrame to analyze (must have 'subreddit' column)
        filter_col: Column name to filter by for stratification

    Example:
        >>> plot_counts_by_community_comparison(
        ...     df=tc_analysis_df,
        ...     filter_col='is_sampled_for_subreddit'
        ... )
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Full dataset
    full_comm_counts = df['subreddit'].value_counts()
    full_comm_counts = full_comm_counts.sort_values(ascending=False)

    ax1.bar(full_comm_counts.index, full_comm_counts.values, color='#487FC683', alpha=0.8)
    ax1.set_title("SSF-Dataset")
    ax1.set_xlabel("Community")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis='x', rotation=90)

    # Stratified dataset
    stratified_df = df[df[filter_col] == True]
    print("stratified_df shape", stratified_df.shape)
    stratified_comm_counts = stratified_df['subreddit'].value_counts()
    stratified_comm_counts = stratified_comm_counts.sort_values(ascending=False)

    ax2.bar(stratified_comm_counts.index, stratified_comm_counts.values, color='#e18719ff', alpha=0.8)
    ax2.set_title("SSF-Dataset-Stratified")
    ax2.set_xlabel("Community")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

def plot_taxonomy_classification_jaccard_iaa_bars(results, output_path):
    """
    Generate bar plot for taxonomy classification inter-annotator agreement (Jaccard index).

    Creates a bar chart showing Jaccard indices for each dimension plus an overall average.
    Dimensions are sorted by score in descending order. Uses black/white styling with
    hatched patterns for dimensions and a solid black bar for the overall score.

    Args:
        results: Dictionary mapping dimension names to their Jaccard index scores
        output_path: Path where the plot should be saved (e.g., 'results/jaccard_iaa.pdf')

    Example:
        >>> results = {'moral': 0.75, 'genre': 0.82, 'intent': 0.68}
        >>> plot_taxonomy_classification_jaccard_iaa_bars(results, 'results/iaa.pdf')
    """
    def format_dimension_name(dimension):
        """Format dimension name for display in plot."""
        return dimension.replace('_', ' ').title()

    # Set up the plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # Calculate overall average
    overall_avg = sum(results.values()) / len(results) if results else 0

    # Get dimensions and sort by score (descending)
    dim_scores = [(dim, results[dim]) for dim in results.keys()]
    dim_scores_sorted = sorted(dim_scores, key=lambda x: x[1], reverse=True)

    # Prepare data with Overall first, then dimensions in descending order
    dimensions = ['overall'] + [dim for dim, _ in dim_scores_sorted]
    scores = [overall_avg] + [score for _, score in dim_scores_sorted]

    # Format dimension names
    formatted_dims = ['Overall'] + [format_dimension_name(dim) for dim, _ in dim_scores_sorted]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(dimensions))
    bars = ax.bar(x, scores, color='white', edgecolor='black', width=0.6, hatch='///', linewidth=1.5)

    # Make the overall bar solid (first bar)
    bars[0].set_hatch(None)
    bars[0].set_facecolor('black')
    bars[0].set_edgecolor('black')

    # Customize the plot
    ax.set_ylabel('Jaccard Index', fontsize=14)
    ax.set_xlabel('Dimension', fontsize=14)
    ax.set_title('Taxonomy Classification Inter-Annotator Agreement', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(formatted_dims, rotation=45, ha='right', va='top', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Set y-axis to start at 0 and go to 1.0 (Jaccard index range)
    ax.set_ylim(0, 1.0)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Bar plot saved to: {output_path}")
