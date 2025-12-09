"""
Similarity Accuracy Computation

This script computes the accuracy of SSF similarity and semantic similarity on preference data
and computes inter-annotator agreement between two annotators.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from ssf.Constants import *
from ssf.Configs import load_config

config = load_config(REPLICATION_CONFIG_PATH)


def get_majority_vote(row, preference_cols):
    """Get majority vote from the 5 preference columns"""
    votes = row[preference_cols].values
    # Filter out NaN values
    valid_votes = [v for v in votes if pd.notna(v)]
    if len(valid_votes) == 0:
        return None
    counter = Counter(valid_votes)
    majority = counter.most_common(1)[0][0]
    return majority


def compute_predictions(df_clean):
    """Compute predictions based on similarity metrics"""
    # SSF similarity prediction
    df_clean['ssf_sim_pred'] = df_clean.apply(
        lambda row: 'A' if row['p1_ssf_sim'] > row['p2_ssf_sim'] else 'B',
        axis=1
    )

    # Semantic similarity prediction
    df_clean['sem_sim_pred'] = df_clean.apply(
        lambda row: 'A' if row['p1_sem_sim'] > row['p2_sem_sim'] else 'B',
        axis=1
    )
    return df_clean


def main():
    # Load data for both annotators
    print("Loading data...")
    df_ann1 = pd.read_csv(f'{config.dirs.data.ssf_sim_global_validation}/ann1/similarity_review_100_rows.csv')
    df_ann2 = pd.read_csv(f'{config.dirs.data.ssf_sim_global_validation}/ann2/similarity_review_100_rows.csv')

    # Get the preference columns (last 5 columns)
    preference_cols = df_ann1.columns[-5:]
    print(f"Preference columns: {list(preference_cols)}")
    print(f"\nAnnotator 1 dataset shape: {df_ann1.shape}")
    print(f"Annotator 2 dataset shape: {df_ann2.shape}")

    # Compute majority votes for both annotators
    print("\n" + "="*60)
    print("COMPUTING MAJORITY VOTES")
    print("="*60)

    df_ann1['majority_preference'] = df_ann1.apply(
        lambda row: get_majority_vote(row, preference_cols), axis=1
    )
    df_ann2['majority_preference'] = df_ann2.apply(
        lambda row: get_majority_vote(row, preference_cols), axis=1
    )

    # Remove rows with no majority preference
    df_ann1_clean = df_ann1[df_ann1['majority_preference'].notna()].copy()
    df_ann2_clean = df_ann2[df_ann2['majority_preference'].notna()].copy()

    print(f"\nAnnotator 1:")
    print(f"  Total rows: {len(df_ann1)}")
    print(f"  Rows with valid majority preference: {len(df_ann1_clean)}")
    print(f"  Majority preference distribution:")
    print(f"  {df_ann1_clean['majority_preference'].value_counts().to_dict()}")

    print(f"\nAnnotator 2:")
    print(f"  Total rows: {len(df_ann2)}")
    print(f"  Rows with valid majority preference: {len(df_ann2_clean)}")
    print(f"  Majority preference distribution:")
    print(f"  {df_ann2_clean['majority_preference'].value_counts().to_dict()}")

    # Generate predictions
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)

    df_ann1_clean = compute_predictions(df_ann1_clean)
    df_ann2_clean = compute_predictions(df_ann2_clean)

    # Compute accuracy for both annotators
    print("\n" + "="*60)
    print("ANNOTATOR 1 ACCURACY RESULTS")
    print("="*60)
    ssf_accuracy_ann1 = (df_ann1_clean['ssf_sim_pred'] == df_ann1_clean['majority_preference']).mean()
    sem_accuracy_ann1 = (df_ann1_clean['sem_sim_pred'] == df_ann1_clean['majority_preference']).mean()
    print(f"SSF Similarity Accuracy: {ssf_accuracy_ann1:.4f} ({ssf_accuracy_ann1*100:.2f}%)")
    print(f"Semantic Similarity Accuracy: {sem_accuracy_ann1:.4f} ({sem_accuracy_ann1*100:.2f}%)")

    print("\n" + "="*60)
    print("ANNOTATOR 2 ACCURACY RESULTS")
    print("="*60)
    ssf_accuracy_ann2 = (df_ann2_clean['ssf_sim_pred'] == df_ann2_clean['majority_preference']).mean()
    sem_accuracy_ann2 = (df_ann2_clean['sem_sim_pred'] == df_ann2_clean['majority_preference']).mean()
    print(f"SSF Similarity Accuracy: {ssf_accuracy_ann2:.4f} ({ssf_accuracy_ann2*100:.2f}%)")
    print(f"Semantic Similarity Accuracy: {sem_accuracy_ann2:.4f} ({sem_accuracy_ann2*100:.2f}%)")

    # Compute IAA
    print("\n" + "="*60)
    print("INTER-ANNOTATOR AGREEMENT (IAA)")
    print("="*60)

    # Match on story pair IDs instead of row index since ann2 may have skipped some rows
    # All four IDs must match: p1_a_id, p1_b_id, p2_a_id, p2_b_id
    merge_keys = ['p1_a_id', 'p1_b_id', 'p2_a_id', 'p2_b_id']

    # Merge on story pair IDs to get only overlapping rows with valid preferences
    merged = df_ann1_clean[merge_keys + ['majority_preference']].merge(
        df_ann2_clean[merge_keys + ['majority_preference']],
        on=merge_keys,
        suffixes=('_ann1', '_ann2')
    )

    print(f"\nNumber of overlapping rows with valid preferences: {len(merged)}")
    print(f"\nAnnotator 1 distribution in overlap:")
    print(merged['majority_preference_ann1'].value_counts())
    print(f"\nAnnotator 2 distribution in overlap:")
    print(merged['majority_preference_ann2'].value_counts())

    # Compute Cohen's Kappa
    kappa = cohen_kappa_score(merged['majority_preference_ann1'], merged['majority_preference_ann2'])

    # Compute simple agreement
    agreement = (merged['majority_preference_ann1'] == merged['majority_preference_ann2']).mean()

    print(f"\n{'='*60}")
    print(f"INTER-ANNOTATOR AGREEMENT METRICS")
    print(f"{'='*60}")
    print(f"Percent Agreement: {agreement:.4f} ({agreement*100:.2f}%)")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"{'='*60}")

    # Show confusion matrix for annotator agreement
    print(f"\nAnnotator Agreement Confusion Matrix:")
    print(pd.crosstab(merged['majority_preference_ann1'], merged['majority_preference_ann2'],
                      rownames=['Annotator 1'], colnames=['Annotator 2'], margins=True))

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save detailed results for both annotators
    output_df_ann1 = df_ann1_clean[['p1_ssf_sim', 'p2_ssf_sim', 'p1_sem_sim', 'p2_sem_sim',
                                    'majority_preference', 'ssf_sim_pred', 'sem_sim_pred']].copy()
    output_df_ann1['ssf_correct'] = output_df_ann1['ssf_sim_pred'] == output_df_ann1['majority_preference']
    output_df_ann1['sem_correct'] = output_df_ann1['sem_sim_pred'] == output_df_ann1['majority_preference']

    output_df_ann2 = df_ann2_clean[['p1_ssf_sim', 'p2_ssf_sim', 'p1_sem_sim', 'p2_sem_sim',
                                    'majority_preference', 'ssf_sim_pred', 'sem_sim_pred']].copy()
    output_df_ann2['ssf_correct'] = output_df_ann2['ssf_sim_pred'] == output_df_ann2['majority_preference']
    output_df_ann2['sem_correct'] = output_df_ann2['sem_sim_pred'] == output_df_ann2['majority_preference']

    # Save annotator-specific results
    output_path_ann1 = f'{SIMILARITY_RESULTS_DIR}/similarity_accuracy_ann1_results.csv'
    output_path_ann2 = f'{SIMILARITY_RESULTS_DIR}/similarity_accuracy_ann2_results.csv'
    output_df_ann1.to_csv(output_path_ann1, index=False)
    output_df_ann2.to_csv(output_path_ann2, index=False)

    print(f"Annotator 1 detailed results saved to: {output_path_ann1}")
    print(f"Annotator 2 detailed results saved to: {output_path_ann2}")

    # Save IAA results
    iaa_output = merged[merge_keys + ['majority_preference_ann1', 'majority_preference_ann2']].copy()
    iaa_output['agreement'] = iaa_output['majority_preference_ann1'] == iaa_output['majority_preference_ann2']
    iaa_output_path = f'{SIMILARITY_RESULTS_DIR}/similarity_iaa_results.csv'
    iaa_output.to_csv(iaa_output_path, index=False)
    print(f"IAA results saved to: {iaa_output_path}")

    # Create comparison plot
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOT")
    print("="*60)

    # Set up the plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # Data for plotting
    metrics = ['SSF-Sim', 'Semantic-Sim']
    ann1_accuracies = [ssf_accuracy_ann1 * 100, sem_accuracy_ann1 * 100]
    ann2_accuracies = [ssf_accuracy_ann2 * 100, sem_accuracy_ann2 * 100]

    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bars - solid for ann1, hatched for ann2
    bars1 = ax.bar(x - width/2, ann1_accuracies, width,
                   label=f'Ann1 (N={len(df_ann1_clean)})', color='black', edgecolor='black')
    bars2 = ax.bar(x + width/2, ann2_accuracies, width,
                   label=f'Ann2 (N={len(df_ann2_clean)})', color='white', edgecolor='black',
                   hatch='///', linewidth=1.5)

    # Customize the plot
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xlabel('Similarity Metric', fontsize=14)
    ax.set_title('Similarity Metric Preference Accuracy by Annotator', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 100)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(fontsize=12, frameon=True, edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save the plot
    plot_output_path = f'{SIMILARITY_RESULTS_DIR}/similarity_accuracy_comparison.pdf'
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_output_path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
