import pandas as pd
import numpy as np
from typing import Any
from .CorpusSplitter import CorpusSplitter


class SsfCorpusSplitter(CorpusSplitter):
    """
    SSF-specific corpus splitter implementation with the following requirements:
    - 5 unseen subreddits for validation (7 rows each = 35 total)
    - 5 unseen subreddits for testing (7 rows each = 35 total)
    - 40 seen subreddits for train/val/test
    - Each seen subreddit contributes:
      - 31 rows to train (40 * 31 = 1240 total)
      - 7 rows to val (40 * 7 = 280 total)
      - 7 rows to test (40 * 7 = 280 total)
    - overall dataset size = 1240 + 280 + 280 + 35 + 35 = 1870
    """

    def add_split_column(self, df: pd.DataFrame, random_state: Any, min_stories_per_community: int = 45) -> tuple[pd.DataFrame, list, list]:
        """
        Add train/val/test split column with SSF-specific requirements.

        Args:
            df: DataFrame to add splits to
            random_state: Random state for reproducibility
            min_stories_per_community: Minimum number of stories required per community

        Returns:
            tuple: (modified_df, val_unseen_subreddits, test_unseen_subreddits)
        """
        df = df.copy()
        df['split'] = None

        # Get qualified subreddits
        subreddit_counts = df['meta.subreddit'].value_counts()
        qualified_subreddits = subreddit_counts[subreddit_counts >= min_stories_per_community].index.tolist()

        if len(qualified_subreddits) < 50:
            raise ValueError(f"Need at least 50 qualified subreddits, found {len(qualified_subreddits)}")

        # Shuffle the subreddits
        np.random.shuffle(qualified_subreddits)

        # Split subreddits into unseen and seen groups
        val_unseen_subreddits = qualified_subreddits[:5]
        test_unseen_subreddits = qualified_subreddits[5:10]
        seen_subreddits = qualified_subreddits[10:50]

        # Process unseen subreddits
        for subreddit in val_unseen_subreddits:
            val_rows = df[df['meta.subreddit'] == subreddit].sample(7, random_state=random_state)
            df.loc[val_rows.index, 'split'] = 'val'

        for subreddit in test_unseen_subreddits:
            test_rows = df[df['meta.subreddit'] == subreddit].sample(7, random_state=random_state)
            df.loc[test_rows.index, 'split'] = 'test'

        # Process seen subreddits
        for subreddit in seen_subreddits:
            subreddit_rows = df[df['meta.subreddit'] == subreddit]

            # Skip if insufficient rows
            if len(subreddit_rows) < min_stories_per_community:
                continue

            # Get all indices and randomly shuffle them
            indices = subreddit_rows.index.tolist()
            np.random.shuffle(indices)

            # Take first 31 for train, next 7 for val, next 7 for test
            train_indices = indices[:31]
            val_indices = indices[31:38]
            test_indices = indices[38:45]

            # Assign splits
            df.loc[train_indices, 'split'] = 'train'
            df.loc[val_indices, 'split'] = 'val'
            df.loc[test_indices, 'split'] = 'test'

        # Summarize the splits
        print(f"Training examples: {(df['split'] == 'train').sum()}")
        print(f"Validation examples: {(df['split'] == 'val').sum()}")
        print(f"Testing examples: {(df['split'] == 'test').sum()}")

        return df, list(val_unseen_subreddits), list(test_unseen_subreddits)
