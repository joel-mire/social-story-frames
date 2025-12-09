"""
Upload SSF corpus to HuggingFace Hub as a private dataset.

Usage:
    # First login to HuggingFace
    huggingface-cli login

    # Then upload dataset
    cd scripts
    python upload_ssf_corpus_to_hf.py --hf_username YOUR_HF_USERNAME

    # Dry run (preview without uploading)
    python upload_ssf_corpus_to_hf.py --hf_username YOUR_HF_USERNAME --dry-run

Requirements:
    - datasets (pip install datasets)
    - huggingface_hub (pip install huggingface_hub)
    - You must be logged in: huggingface-cli login
"""

import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import sys
from ast import literal_eval

from ssf.Configs import load_config
from ssf.Constants import *
from ssf.Taxonomy import Taxonomy
from ssf.utils.TaxonomyDataUtils import get_cats_col_name


def build_column_mapping(taxonomy: Taxonomy, prompt_suffix: str = PROMPT_COL_SUFFIX_FULL_CONTEXT) -> dict:
    """
    Build column mapping programmatically from taxonomy and constants.

    Args:
        taxonomy: Taxonomy object with dimension information
        prompt_suffix: Prompt column suffix (default uses full context)

    Returns:
        Dict mapping new column names to original CSV column names
    """
    # Core metadata columns
    column_mapping = {
        'id': 'id',
        'speaker': 'speaker',
        'conversation_id': 'conversation_id',
        'split': 'split',
        'community': f'meta.{COMMUNITY_META_KEY}',
        'score': 'meta.score',
        'ancestralUttIds': f'meta.{ANCESTRAL_UTT_IDS_META_KEY}',
        'prevUttIds': f'meta.{PREV_UTT_IDS_META_KEY}',
        'storySeeker': 'meta.storySeeker',
        'perspective': f'meta.{PERSPECTIVE_META_KEY}',
        'summary': f'meta.{SUMMARY_META_KEY}',
        'progenitorContext': f'meta.{PROGENITOR_CONTEXT_META_KEY}',
        'conversationContext': f'meta.{CONVERSATION_CONTEXT_META_KEY}',
        'communityDescription': f'meta.{COMMUNITY_DESCRIPTION_META_KEY}',
        'communityValues': f'meta.{COMMUNITY_VALUES_META_KEY}',
    }

    # Add dimension inferences and labels programmatically
    for dim in taxonomy.get_dims():
        # Inference columns (gen0)
        column_mapping[f'{dim}_inference'] = f'{prompt_suffix}${dim}_gen0'

        # Classification/label columns (gen0_cats)
        column_mapping[f'{dim}_labels'] = f'{prompt_suffix}${dim}_gen0_cats'

    return column_mapping


def build_csv_converters(taxonomy: Taxonomy, prompt_suffix: str = PROMPT_COL_SUFFIX_FULL_CONTEXT) -> dict:
    """
    Build converters for pd.read_csv to parse string lists as actual lists.

    This ensures that columns containing string representations of lists
    (e.g., "['item1', 'item2']") are converted to actual Python lists.

    Args:
        taxonomy: Taxonomy object with dimension information
        prompt_suffix: Prompt column suffix (default uses full context)

    Returns:
        Dict mapping column names to converter functions for pd.read_csv
    """
    converters = {}

    # Add converters for dimension label columns (gen0_cats)
    for dim in taxonomy.get_dims():
        cats_col = get_cats_col_name(prompt_suffix, dim, "gen0")
        converters[cats_col] = literal_eval

    # Add converters for meta list columns
    converters[f'meta.{ANCESTRAL_UTT_IDS_META_KEY}'] = literal_eval
    converters[f'meta.{PREV_UTT_IDS_META_KEY}'] = literal_eval

    return converters


def upload_ssf_corpus_to_hf(
    csv_path: str,
    dataset_name: str,
    column_mapping: dict,
    csv_converters: dict,
    hf_username: str = None,
    private: bool = True,
    dry_run: bool = False
):
    """
    Upload SSF CSV to HuggingFace with automatic split handling.

    Args:
        csv_path: Path to the ssf.csv file
        dataset_name: Name for the HuggingFace dataset
        column_mapping: Dict mapping new column names to original column names
        csv_converters: Dict mapping column names to converter functions for parsing lists
        hf_username: HuggingFace username (if None, will use authenticated user)
        private: Whether to make the dataset private
        dry_run: If True, don't actually upload, just show what would happen
    """
    print("=" * 80)
    print(f"HuggingFace Dataset Upload: {dataset_name}")
    print("=" * 80)

    # Check if file exists
    if not Path(csv_path).exists():
        print(f"[ERROR] File not found: {csv_path}")
        return False

    # Load the CSV with converters to parse list columns
    print(f"\n1. Loading CSV: {csv_path}")
    print(f"   Using converters for {len(csv_converters)} list columns")
    df = pd.read_csv(csv_path, converters=csv_converters, low_memory=False)
    print(f"   Total rows: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")

    # Check for split column
    if 'split' not in df.columns:
        print("   [ERROR] 'split' column not found in CSV!")
        return False

    # Show split distribution
    split_counts = df['split'].value_counts()
    print(f"\n   Split distribution:")
    for split_name, count in split_counts.items():
        print(f"      {split_name}: {count} rows")
    nan_count = df['split'].isna().sum()
    if nan_count > 0:
        print(f"      NaN: {nan_count} rows")

    # Select and rename columns
    print(f"\n2. Selecting and renaming columns")
    print(f"   Requested columns: {len(column_mapping)}")

    # Check which columns are available
    missing_cols = []
    available_mapping = {}

    for new_name, orig_name in column_mapping.items():
        if orig_name in df.columns:
            available_mapping[new_name] = orig_name
        else:
            missing_cols.append(f"{new_name} (from {orig_name})")

    if missing_cols:
        print(f"\n   [WARNING] {len(missing_cols)} requested columns not found:")
        for col in missing_cols[:5]:
            print(f"      - {col}")
        if len(missing_cols) > 5:
            print(f"      ... and {len(missing_cols) - 5} more")

    # Select original columns and rename
    orig_cols = list(available_mapping.values())
    df_selected = df[orig_cols].copy()

    # Create reverse mapping for rename
    rename_map = {orig: new for new, orig in available_mapping.items()}
    df_selected = df_selected.rename(columns=rename_map)

    # Replace sentinel '<<<NO_OP>>>' strings with empty strings
    print(f"\n   Replacing sentinel '<<<NO_OP>>>' with empty strings...")
    sentinel_count = (df_selected == '<<<NO_OP>>>').sum().sum()
    df_selected = df_selected.replace('<<<NO_OP>>>', '')
    print(f"      Replaced {sentinel_count} occurrences")

    print(f"\n   Final dataset:")
    print(f"      Rows: {len(df_selected)}")
    print(f"      Columns: {len(df_selected.columns)}")

    # Show column categories
    core_cols = [c for c in df_selected.columns
                 if not c.endswith('_inference') and not c.endswith('_labels') and c != 'split']
    inference_cols = [c for c in df_selected.columns if c.endswith('_inference')]
    label_cols = [c for c in df_selected.columns if c.endswith('_labels')]

    print(f"\n   Column breakdown:")
    print(f"      Core metadata: {len(core_cols)} columns")
    print(f"      Dimension inferences: {len(inference_cols)} columns")
    print(f"      Dimension labels: {len(label_cols)} columns")

    if dry_run:
        print(f"\n   All columns ({len(df_selected.columns)}):")
        for col in df_selected.columns:
            print(f"      - {col}")
    else:
        print(f"\n   Sample columns:")
        for i, col in enumerate(df_selected.columns):
            if i < 10:
                print(f"      - {col}")
        if len(df_selected.columns) > 10:
            print(f"      ... and {len(df_selected.columns) - 10} more")

    # Create DatasetDict with splits
    print(f"\n3. Creating DatasetDict with splits")
    dataset_dict = {}

    # Create splits in desired order: full, train, val, test

    # 1. Create "full" split with ALL rows (including NaN splits), keeping the split column
    full_df = df_selected.copy()
    nan_count = full_df['split'].isna().sum()
    labeled_count = len(full_df) - nan_count

    dataset_dict['full'] = Dataset.from_pandas(full_df, preserve_index=False)
    print(f"   Created 'full' split: {len(full_df)} rows, {len(full_df.columns)} columns")
    print(f"      - Rows with splits (train/val/test): {labeled_count}")
    print(f"      - Rows without splits (NaN): {nan_count}")

    # 2. Create individual splits (train, val, test) in order
    # Keep the split column so all splits have the same schema
    for split_name in ['train', 'val', 'test']:
        # Only create if this split exists in the data
        if split_name not in df_selected['split'].values:
            continue

        split_df = df_selected[df_selected['split'] == split_name].copy()

        dataset_dict[split_name] = Dataset.from_pandas(split_df, preserve_index=False)
        print(f"   Created '{split_name}' split: {len(split_df)} rows, {len(split_df.columns)} columns")

    dataset = DatasetDict(dataset_dict)

    # Print dataset info
    print(f"\n   Dataset structure:")
    print(dataset)

    # Determine repository name
    if hf_username:
        repo_id = f"{hf_username}/{dataset_name}"
    else:
        # Will use the authenticated user's username
        repo_id = dataset_name
        print(f"\n   [NOTE] Using authenticated user's HuggingFace account")

    print(f"\n4. Upload Configuration:")
    print(f"   Repository: {repo_id}")
    print(f"   Private: {private}")
    print(f"   Total rows: {len(df_selected)}")
    print(f"   Total columns: {len(df_selected.columns)}")

    if dry_run:
        print(f"\n   [DRY RUN] Not uploading")
        print(f"\n   Would upload to: https://huggingface.co/datasets/{repo_id}")
        return True

    # Upload to HuggingFace
    print(f"\n5. Uploading to HuggingFace...")
    try:
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=True  # Use token from huggingface-cli login
        )
        print(f"\n   [SUCCESS] Uploaded to HuggingFace!")
        print(f"\n   Dataset URL: https://huggingface.co/datasets/{repo_id}")
        return True

    except Exception as e:
        print(f"\n   [ERROR] Upload failed: {e}")
        print(f"\n   Make sure you're logged in with: huggingface-cli login")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload SSF corpus to HuggingFace as a private dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uploads ssf.csv with curated columns including:
- Core metadata (17 columns): id, speaker, conversation_id, split, community, score, etc.
- Dimension inferences (10 columns): overall_goal_inference, narrative_intent_inference, etc.
- Dimension labels (10 columns): overall_goal_labels, narrative_intent_labels, etc.

Splits (all splits have 37 columns with matching schema):
- full: Contains ALL rows (6,140 rows) - includes rows with and without split labels
- train: Contains only train split (1,184 rows)
- val: Contains only validation split (297 rows)
- test: Contains only test split (297 rows)

Examples:
  # Upload with default settings (private)
  python upload_ssf_corpus_to_hf.py --hf_username YOUR_USERNAME

  # Dry run to see what would be uploaded
  python upload_ssf_corpus_to_hf.py --hf_username YOUR_USERNAME --dry-run

  # Make dataset public
  python upload_ssf_corpus_to_hf.py --hf_username YOUR_USERNAME --public

  # Custom dataset name
  python upload_ssf_corpus_to_hf.py --hf_username YOUR_USERNAME --name my-ssf-corpus
        """
    )

    parser.add_argument(
        '--hf_username',
        type=str,
        required=True,
        help='HuggingFace username'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=REPLICATION_CONFIG_PATH,
        help='Config path (default: ../configs/replication.yaml)'
    )

    parser.add_argument(
        '--name',
        default="ssf-corpus",
        help='Name for the HuggingFace dataset (default: ssf-corpus)'
    )

    parser.add_argument(
        '--public',
        action='store_true',
        help='Make the dataset public (default: private)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without uploading'
    )

    args = parser.parse_args()

    print("SSF Corpus Upload to HuggingFace Hub")
    print("=" * 80)

    # Load config
    print(f"\nLoading config: {args.config}")
    config = load_config(args.config)
    print(f"Config ID: {config.id}")

    # Initialize taxonomy
    print(f"\nLoading taxonomy: {TAXONOMY_DIR}")
    taxonomy = Taxonomy(taxonomy_dir=TAXONOMY_DIR)
    dims = taxonomy.get_dims()
    print(f"Taxonomy dimensions: {len(dims)}")
    print(f"  {', '.join(dims)}")

    # Build column mapping programmatically
    print(f"\nBuilding column mapping...")
    column_mapping = build_column_mapping(taxonomy, PROMPT_COL_SUFFIX_FULL_CONTEXT)
    print(f"  Total columns: {len(column_mapping)}")

    # Build CSV converters for list columns
    print(f"\nBuilding CSV converters...")
    csv_converters = build_csv_converters(taxonomy, PROMPT_COL_SUFFIX_FULL_CONTEXT)
    print(f"  Converters for {len(csv_converters)} list columns")

    # Get CSV path from config
    csv_path = f"{config.dirs.data.corpus}/ssf.csv"
    print(f"\nCorpus path: {csv_path}")

    # Upload
    success = upload_ssf_corpus_to_hf(
        csv_path=csv_path,
        dataset_name=args.name,
        column_mapping=column_mapping,
        csv_converters=csv_converters,
        hf_username=args.hf_username,
        private=not args.public,
        dry_run=args.dry_run
    )

    # Summary
    print(f"\n{'=' * 80}")
    if success:
        print(f"[SUCCESS] Dataset {'would be' if args.dry_run else 'has been'} uploaded")
        if args.dry_run:
            print("[DRY RUN] Remove --dry-run to upload")
    else:
        print(f"[FAILED] Upload did not complete")
    print("=" * 80)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
