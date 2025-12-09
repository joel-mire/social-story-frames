#!/usr/bin/env python3
"""
Context Summarization Annotation Analysis and Comparison

This script analyzes context summarization annotations from two annotators:
1. Computes descriptive statistics (mean, std) for each annotator
2. Generates comparison bar plots
3. Calculates inter-rater agreement metrics (IAA):
   - Raw agreement percentage (exact matches)
   - Agreement within 1 point
   - Brennan-Prediger coefficient (with ordinal weights)
   - Gwet's AC2 coefficient (with ordinal weights)
   - Cohen's kappa (with quadratic weights)
4. Generates LaTeX table with IAA results

Outputs:
- Bar plot: ctx_sum_ann_comparison.pdf
- LaTeX table: context_summarization_iaa_table.tex
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from irrCAC.raw import CAC

from ssf.Constants import CONTEXT_SUMMARIZATION_RESULTS_DIR, REPLICATION_CONFIG_PATH
from ssf.Configs import load_config
from ssf.utils import ResultsVisUtils

config = load_config(REPLICATION_CONFIG_PATH)


# Configuration
DATA_DIR = f"{config.dirs.data.annotations}/ctx_summarization"

# Mapping of file names to readable summary type labels
SUMMARY_TYPE_MAPPING = {
    "_subreddit_description_human_ann.csv": "subreddit purpose",
    "_subreddit_value_human_ann.csv": "subreddit values/norms",
    "_progenitor_summary_human_ann.csv": "initial post",
    "_conversation_summary_human_ann.csv": "conversation history"
}


def calculate_within_n_agreement(ratings1, ratings2, n=1):
    """Calculate percentage of ratings that are within n points of each other."""
    differences = [abs(r1 - r2) for r1, r2 in zip(ratings1, ratings2)]
    within_n = sum(1 for diff in differences if diff <= n)
    return (within_n / len(differences)) * 100


def load_and_prepare_data(file_path, n_rows=30):
    """Load CSV file and prepare data for analysis."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean column names
    return df.head(n_rows)


def collect_annotator_stats():
    """Collect descriptive statistics for both annotators."""
    ann1_dir = f"{DATA_DIR}/ann1"
    ann2_dir = f"{DATA_DIR}/ann2"

    ann1_stats = {}
    ann2_stats = {}

    for filename, summary_type in SUMMARY_TYPE_MAPPING.items():
        ann1_file = f"{ann1_dir}/{filename}"
        ann2_file = f"{ann2_dir}/{filename}"

        # Load annotator 1 data
        ann1_df = load_and_prepare_data(ann1_file)
        ann1_consistency = ann1_df['Consistency_0'].tolist()
        ann1_relevance = ann1_df['Relevance_0'].tolist()

        ann1_stats[summary_type] = {
            'consistency_mean': np.mean(ann1_consistency),
            'consistency_std': np.std(ann1_consistency),
            'relevance_mean': np.mean(ann1_relevance),
            'relevance_std': np.std(ann1_relevance)
        }

        # Load annotator 2 data
        ann2_df = load_and_prepare_data(ann2_file)
        ann2_consistency = ann2_df['Consistency_0'].tolist()
        ann2_relevance = ann2_df['Relevance_0'].tolist()

        ann2_stats[summary_type] = {
            'consistency_mean': np.mean(ann2_consistency),
            'consistency_std': np.std(ann2_consistency),
            'relevance_mean': np.mean(ann2_relevance),
            'relevance_std': np.std(ann2_relevance)
        }

    return ann1_stats, ann2_stats


def calculate_dimension_metrics(ann1_ratings, ann2_ratings):
    """Calculate all agreement metrics for a single dimension (consistency or relevance)."""
    # Raw agreement metrics
    exact_matches = sum(1 for r1, r2 in zip(ann1_ratings, ann2_ratings) if r1 == r2)
    raw_agreement = (exact_matches / len(ann1_ratings)) * 100
    within1_agreement = calculate_within_n_agreement(ann1_ratings, ann2_ratings, 1)

    # Cohen's kappa with quadratic weights
    kappa_quadratic = cohen_kappa_score(ann1_ratings, ann2_ratings, weights='quadratic')

    # CAC metrics with ordinal weights
    data = pd.DataFrame({'rater1': ann1_ratings, 'rater2': ann2_ratings})
    cac = CAC(data, weights="ordinal")

    # Brennan-Prediger and AC2 coefficients
    bp_result = cac.bp()
    bp_coefficient = bp_result['est']['coefficient_value']
    ac2_result = cac.gwet()
    ac2_coefficient = ac2_result['est']['coefficient_value']

    return {
        'raw_agreement': raw_agreement,
        'within1_agreement': within1_agreement,
        'bp': bp_coefficient,
        'ac2': ac2_coefficient,
        'kappa_quadratic': kappa_quadratic
    }


def calculate_agreement_metrics():
    """Calculate all agreement metrics for all summary types."""
    # Set up file paths
    ann1_dir = f"{DATA_DIR}/ann1"
    ann2_dir = f"{DATA_DIR}/ann2"

    results = {}

    # Process each summary type
    for filename, summary_type in SUMMARY_TYPE_MAPPING.items():
        ann1_file = f"{ann1_dir}/{filename}"
        ann2_file = f"{ann2_dir}/{filename}"

        print(f"  Analyzing: {summary_type} ({filename})")

        try:
            # Load and prepare data
            ann1_df = load_and_prepare_data(ann1_file)
            ann2_df = load_and_prepare_data(ann2_file)

            # Ensure same number of rows
            min_rows = min(len(ann1_df), len(ann2_df))
            ann1_df = ann1_df.head(min_rows)
            ann2_df = ann2_df.head(min_rows)

            # Extract ratings for both dimensions
            ann1_consistency = ann1_df['Consistency_0'].tolist()
            ann1_relevance = ann1_df['Relevance_0'].tolist()
            ann2_consistency = ann2_df['Consistency_0'].tolist()
            ann2_relevance = ann2_df['Relevance_0'].tolist()

            # Calculate metrics for both dimensions
            consistency_metrics = calculate_dimension_metrics(ann1_consistency, ann2_consistency)
            relevance_metrics = calculate_dimension_metrics(ann1_relevance, ann2_relevance)

            # Store results
            results[summary_type] = {
                'consistency': consistency_metrics,
                'relevance': relevance_metrics
            }

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    return results


def format_summary_type_name(summary_type):
    """Format summary type names for display in the table."""
    type_mapping = {
        "subreddit purpose": "Subreddit Purpose",
        "subreddit values/norms": "Subreddit Norms/Values",
        "initial post": "Initial Post",
        "conversation history": "Conversation History"
    }
    return type_mapping.get(summary_type, summary_type.replace("_", " ").title())


def generate_latex_table(results):
    """Generate LaTeX table from agreement analysis results."""
    # Table header
    latex = r"""\begin{table*}[htbp]
\centering
\small
\begin{tabular}{l|ccccc|ccccc}
\hline
\multirow{2}{*}{Summary Type} & \multicolumn{5}{c|}{Consistency} & \multicolumn{5}{c}{Relevance} \\
\cline{2-11}
& \%Agr & \%±1 & $\kappa_b$ & AC$_2$ & $\kappa_c$ & \%Agr & \%±1 & $\kappa_b$ & AC$_2$ & $\kappa_c$ \\
\hline
"""

    # Data rows
    for summary_type, metrics in results.items():
        formatted_type = format_summary_type_name(summary_type)

        # Extract metrics for consistency and relevance
        c = metrics['consistency']
        r = metrics['relevance']

        # Format table row
        latex += (f"{formatted_type} & "
                 f"{c['raw_agreement']:.1f} & {c['within1_agreement']:.1f} & "
                 f"{c['bp']:.2f} & {c['ac2']:.2f} & {c['kappa_quadratic']:.2f} & "
                 f"{r['raw_agreement']:.1f} & {r['within1_agreement']:.1f} & "
                 f"{r['bp']:.2f} & {r['ac2']:.2f} & {r['kappa_quadratic']:.2f} \\\\\n")

    # Table footer with caption
    latex += r"""\hline
\end{tabular}
\caption{Inter-rater Agreement Metrics by Summary Type and Dimension. \%Agr = exact agreement; \%±1 = agreement within 1 point; $\kappa_b$ = Brennan-Prediger (ordinal); AC$_2$ = Gwet's AC$_2$ (ordinal); $\kappa_c$ = Cohen's kappa (quadratic).}
\label{tab:context_summarization_iaa}
\end{table*}"""

    return latex


def main():
    """Main function to run the full analysis."""
    print("=" * 80)
    print("CONTEXT SUMMARIZATION ANNOTATION ANALYSIS")
    print("=" * 80)

    # Ensure results directory exists
    Path(CONTEXT_SUMMARIZATION_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Step 1: Collect descriptive statistics
    print("\n1. Collecting descriptive statistics for both annotators...")
    ann1_stats, ann2_stats = collect_annotator_stats()
    print("   Done!")

    # Step 2: Generate comparison bar plot
    print("\n2. Generating comparison bar plot...")
    plot_path = f'{CONTEXT_SUMMARIZATION_RESULTS_DIR}/ctx_sum_ann_comparison.pdf'
    ResultsVisUtils.plot_ctx_sum_ann_bars(ann1_stats, ann2_stats, plot_path)
    print(f"   Saved to: {plot_path}")

    # Step 3: Calculate IAA metrics
    print("\n3. Calculating inter-rater agreement metrics...")
    iaa_results = calculate_agreement_metrics()

    if not iaa_results:
        print("   No results to process. Check file paths and data availability.")
        return

    print("   Done!")

    # Step 4: Generate LaTeX table
    print("\n4. Generating LaTeX table...")
    latex_table = generate_latex_table(iaa_results)
    table_path = f"{CONTEXT_SUMMARIZATION_RESULTS_DIR}/context_summarization_iaa_table.tex"

    with open(table_path, 'w') as f:
        f.write(latex_table)

    print(f"   Saved to: {table_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - Bar plot: {plot_path}")
    print(f"  - LaTeX table: {table_path}")


if __name__ == "__main__":
    main()
