#!/usr/bin/env python3
"""
Compute Jaccard Index for Inter-rater Agreement in Taxonomy Classification.

This script calculates Jaccard indices for taxonomy classification annotations
across different dimensions, applying appropriate mappings (e.g., moral values)
and generates a LaTeX table with the results.
"""

import pandas as pd
import os
from pathlib import Path
from ssf.Constants import *
from ssf.Taxonomy import Taxonomy
from ssf.helpers.TaxonomyEvaluator import map_moral_values
from ssf.utils.ResultsVisUtils import plot_taxonomy_classification_jaccard_iaa_bars
from ssf.Configs import load_config

config = load_config(REPLICATION_CONFIG_PATH)

DATA_DIR = f"{config.dirs.data.annotations}/tax_class"
N_INSTANCES = 50  # Number of instances to analyze

taxonomy = Taxonomy(TAXONOMY_DIR)

def jaccard_index(set1, set2):
    """Compute Jaccard index between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def extract_labels_per_instance(df, n_instances=None, dimension=None):
    """Extract labels for each instance."""
    if n_instances is None:
        n_instances = N_INSTANCES

    df_subset = df.head(n_instances)
    labels_per_instance = []

    for _, row in df_subset.iterrows():
        if pd.notna(row['labels']):
            # Split comma-separated labels and clean them
            labels = [label.strip() for label in str(row['labels']).split(',')]

            # Apply moral mapping if this is the moral dimension
            if dimension == 'moral':
                labels = map_moral_values(labels)

            labels_per_instance.append(set(labels))
        else:
            labels_per_instance.append(set())

    return labels_per_instance


def get_available_dimensions():
    """Get list of available dimensions in the proper order from taxonomy."""
    # Get dimensions from taxonomy object in the correct order
    taxonomy_dims = taxonomy.get_dims()

    # Filter to only include dimensions that have annotation files
    available_dims = []
    for dim in taxonomy_dims:
        ann1_file = os.path.join(DATA_DIR, f"test_{dim}_hum_ann.csv")
        ann2_file = os.path.join(DATA_DIR, "ann2", f"test_{dim}_hum_ann.csv")
        if os.path.exists(ann1_file) and os.path.exists(ann2_file):
            available_dims.append(dim)

    return available_dims


def calculate_jaccard_for_dimension(dimension):
    """Calculate Jaccard index for a specific dimension."""
    ann1_file = os.path.join(DATA_DIR, f"test_{dimension}_hum_ann.csv")
    ann2_file = os.path.join(DATA_DIR, "ann2", f"test_{dimension}_hum_ann.csv")

    if not (os.path.exists(ann1_file) and os.path.exists(ann2_file)):
        return None

    # Load data
    ann1_df = pd.read_csv(ann1_file)
    ann2_df = pd.read_csv(ann2_file)

    # Extract labels for each annotator
    ann1_labels_per_instance = extract_labels_per_instance(ann1_df, N_INSTANCES, dimension)
    ann2_labels_per_instance = extract_labels_per_instance(ann2_df, N_INSTANCES, dimension)

    # Compute Jaccard scores for each instance
    jaccard_scores = []
    for ann1_inst, ann2_inst in zip(ann1_labels_per_instance, ann2_labels_per_instance):
        jaccard_scores.append(jaccard_index(ann1_inst, ann2_inst))

    # Return average Jaccard index
    return sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0


def calculate_all_jaccard_indices():
    """Calculate Jaccard indices for all available dimensions."""
    dimensions = get_available_dimensions()
    results = {}

    for dim in dimensions:
        print(f"Analyzing dimension: {dim}")
        jaccard_score = calculate_jaccard_for_dimension(dim)

        if jaccard_score is not None:
            results[dim] = jaccard_score
            print(f"  {dim:25}: {jaccard_score:.4f}")
        else:
            print(f"  {dim:25}: Files missing")

    return results


def main():
    """Main function to compute Jaccard indices and generate bar plot."""
    print("Computing Jaccard Index for Taxonomy Classification IAA")
    print("=" * 60)

    # Calculate Jaccard indices for all dimensions
    results = calculate_all_jaccard_indices()

    if not results:
        print("No results to process. Check file paths and data availability.")
        return

    # Generate bar plot
    print("\nGenerating bar plot...")

    # Ensure results directory exists and save plot
    Path(TAX_CLASS_RESULTS_DIR).mkdir(exist_ok=True)
    output_file = f"{TAX_CLASS_RESULTS_DIR}/taxonomy_classification_jaccard_iaa.pdf"

    plot_taxonomy_classification_jaccard_iaa_bars(results, output_file)

    # Print summary
    print("\n" + "=" * 60)
    if results:
        overall_avg = sum(results.values()) / len(results)
        print(f"Overall Average Jaccard Index: {overall_avg:.4f}")
        print(f"Number of dimensions analyzed: {len(results)}")
        print("=" * 60)


if __name__ == "__main__":
    main()