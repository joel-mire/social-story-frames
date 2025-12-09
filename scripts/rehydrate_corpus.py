"""
Rehydrate SSF corpus CSV files by restoring the '_text' column.

This script restores the '_text' column to dehydrated SSF corpus CSVs using
ConvoKit's reddit-corpus-small. The text is retrieved using the story ID.

Reads from: data/replication/corpus/text_masked/text_masked_*.csv
Writes to: data/replication/corpus/*.csv
"""

import pandas as pd
from pathlib import Path
from convokit import Corpus, download
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

# CSV files to rehydrate
TEXT_MASKED_CSV_FILES = [
    f"text_masked_{SSF_DF_PATH}",  # text_masked_ssf.csv
    "text_masked_ssf_split.csv",
    "text_masked_ssf_split_test.csv",
    "text_masked_ssf_split_train.csv",
    "text_masked_ssf_split_val.csv",
    "text_masked_ssf_gen_eval.csv"
]


def load_reddit_corpus():
    """Load the reddit-corpus-small from ConvoKit."""
    logger.info("Loading reddit-corpus-small from ConvoKit...")
    reddit_corpus = Corpus(download('reddit-corpus-small'))
    logger.info(f"Loaded reddit corpus with {len(reddit_corpus.utterances)} utterances")
    return reddit_corpus


def rehydrate_csv(input_path: Path, output_path: Path, reddit_corpus: Corpus) -> None:
    """
    Restore the 'text' column to a dehydrated CSV file.

    Args:
        input_path: Path to the dehydrated CSV file
        output_path: Path to save the rehydrated CSV file
        reddit_corpus: ConvoKit reddit corpus for text retrieval
    """
    logger.info(f"Rehydrating {input_path.name}...")

    # Read the dehydrated CSV
    df = pd.read_csv(input_path)
    logger.info(f"  Input shape: {df.shape}")

    # Check if '_text' column already exists
    if '_text' in df.columns:
        logger.warning(f"  '_text' column already exists in {input_path.name}")
        logger.info(f"  Skipping rehydration")
        return

    # Check if 'id' column exists
    if 'id' not in df.columns:
        logger.error(f"  'id' column not found in {input_path.name}")
        logger.error(f"  Cannot rehydrate without 'id' column")
        return

    # Rehydrate the '_text' column
    logger.info(f"  Retrieving text for {len(df)} stories...")

    def get_text(story_id):
        """Get text for a given story ID from reddit corpus."""
        try:
            utterance = reddit_corpus.get_utterance(story_id)
            return utterance.text
        except Exception as e:
            logger.warning(f"  Failed to get text for ID {story_id}: {e}")
            return None

    df['_text'] = df['id'].apply(get_text)

    # Count how many texts were successfully retrieved
    num_retrieved = df['_text'].notna().sum()
    logger.info(f"  Successfully retrieved {num_retrieved}/{len(df)} texts ({100*num_retrieved/len(df):.1f}%)")

    # Place '_text' column at the end
    cols = df.columns.tolist()
    cols.remove('_text')
    cols.append('_text')
    df = df[cols]

    # Save the rehydrated version
    df.to_csv(output_path, index=False)
    logger.info(f"  Rehydrated shape: {df.shape}")
    logger.info(f"  Saved to {output_path}")
    logger.info("")


def rehydrate_all_files():
    """Rehydrate all text_masked CSV files."""
    # Load reddit corpus once
    reddit_corpus = load_reddit_corpus()
    logger.info("")

    # Process each text_masked CSV file
    for csv_file in TEXT_MASKED_CSV_FILES:
        input_path = TEXT_MASKED_DIR / csv_file

        # Check if file exists
        if not input_path.exists():
            logger.warning(f"Skipping {csv_file} - file not found")
            logger.warning(f"  Expected path: {input_path}")
            continue

        # Create output path (remove "text_masked_" prefix)
        output_filename = csv_file.replace("text_masked_", "")
        output_path = CORPUS_DIR / output_filename

        # Rehydrate the file
        rehydrate_csv(input_path, output_path, reddit_corpus)

    logger.info("All rehydrations complete!")


if __name__ == "__main__":
    rehydrate_all_files()
