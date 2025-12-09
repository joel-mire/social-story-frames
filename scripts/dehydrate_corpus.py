"""
Dehydrate SSF corpus CSV files by removing text columns.

This script creates text_masked versions of the SSF corpus CSVs by removing
the following columns:
- text
- meta.top_level_comment
- meta.original
- Any column containing 'single_output_prompt'

The dehydrated files are saved to data/replication/corpus/text_masked/
"""

import pandas as pd
from pathlib import Path
import logging

from ssf.Constants import REPLICATION_CONFIG_PATH, SSF_DF_PATH
from ssf.Configs import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config(REPLICATION_CONFIG_PATH)

# Define paths
CORPUS_DIR = Path(config.dirs.data.corpus).resolve()
TEXT_MASKED_DIR = CORPUS_DIR / "text_masked"

# CSV files to dehydrate
CSV_FILES = [
    SSF_DF_PATH,  # ssf.csv
    "ssf_split.csv",
    "ssf_split_test.csv",
    "ssf_split_train.csv",
    "ssf_split_val.csv",
    "ssf_gen_eval.csv"
]

# Columns to exclude (exact matches)
COLUMNS_TO_EXCLUDE = [
    "text",
    "_text",
    "meta.top_level_comment",
    "meta.original"
]

# Column patterns to exclude (any column containing these strings)
COLUMN_PATTERNS_TO_EXCLUDE = [
    "single_output_prompt"
]


def dehydrate_csv(input_path: Path, output_path: Path) -> None:
    """
    Remove text columns from a CSV file and save the result.

    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the dehydrated CSV file
    """
    logger.info(f"Dehydrating {input_path.name}...")

    # Read the CSV
    df = pd.read_csv(input_path)
    logger.info(f"  Original shape: {df.shape}")
    logger.info(f"  Original columns: {len(df.columns)}")

    # Find columns to drop by exact name
    columns_to_drop = [col for col in COLUMNS_TO_EXCLUDE if col in df.columns]

    # Find columns to drop by pattern matching
    pattern_columns_to_drop = [
        col for col in df.columns
        if any(pattern in col for pattern in COLUMN_PATTERNS_TO_EXCLUDE)
    ]

    # Combine both lists
    columns_to_drop = list(set(columns_to_drop + pattern_columns_to_drop))

    logger.info(f"  Dropping {len(columns_to_drop)} columns:")
    for col in sorted(columns_to_drop):
        logger.info(f"    - {col}")

    # Drop the columns
    df_dehydrated = df.drop(columns=columns_to_drop)
    logger.info(f"  Dehydrated shape: {df_dehydrated.shape}")
    logger.info(f"  Dehydrated columns: {len(df_dehydrated.columns)}")

    # Save the dehydrated version
    df_dehydrated.to_csv(output_path, index=False)
    logger.info(f"  Saved to {output_path}")
    logger.info("")


def main():
    """Dehydrate all SSF corpus CSV files."""
    # Create text_masked directory if it doesn't exist
    TEXT_MASKED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created text_masked directory: {TEXT_MASKED_DIR}")
    logger.info("")

    # Process each CSV file
    for csv_file in CSV_FILES:
        input_path = CORPUS_DIR / csv_file

        # Check if file exists
        if not input_path.exists():
            logger.warning(f"Skipping {csv_file} - file not found")
            logger.warning(f"  Expected path: {input_path}")
            continue

        # Create output path with "text_masked_" prefix
        output_filename = f"text_masked_{csv_file}"
        output_path = TEXT_MASKED_DIR / output_filename

        # Dehydrate the file
        dehydrate_csv(input_path, output_path)

    logger.info("Dehydration complete!")


if __name__ == "__main__":
    main()
