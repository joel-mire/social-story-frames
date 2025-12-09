import pandas as pd
from typing import Any
from pathlib import Path
from .CorpusSplitter import CorpusSplitter


class PreassignedSplitCorpusSplitter(CorpusSplitter):
    """
    Corpus splitter that loads pre-existing split assignments from CSV files.

    This is useful when you have pre-existing annotations or splits that you want to preserve.
    """

    def __init__(self, train_split_path: str | Path, test_split_path: str | Path, val_split_path: str | Path, id_column: str = 'id'):
        """
        Initialize the PreassignedSplitCorpusSplitter.

        Args:
            train_split_path: Path to CSV file containing train split instances
            test_split_path: Path to CSV file containing test split instances
            val_split_path: Path to CSV file containing validation split instances
            id_column: Name of the column containing instance IDs (default: 'id')
        """
        self.train_split_path = Path(train_split_path)
        self.test_split_path = Path(test_split_path)
        self.val_split_path = Path(val_split_path)
        self.id_column = id_column

        # Validate that files exist
        if not self.train_split_path.exists():
            raise FileNotFoundError(f"Train split file not found: {self.train_split_path}")
        if not self.test_split_path.exists():
            raise FileNotFoundError(f"Test split file not found: {self.test_split_path}")
        if not self.val_split_path.exists():
            raise FileNotFoundError(f"Validation split file not found: {self.val_split_path}")

    def add_split_column(self, df: pd.DataFrame, random_state: Any = None, **kwargs) -> tuple[pd.DataFrame, dict, dict, dict]:
        """
        Add train/val/test split column based on pre-existing split assignments.
        Returns data in the order defined by the authoritative split files (train, val, test).

        Args:
            df: DataFrame to add splits to
            random_state: Not used (kept for interface compatibility)
            **kwargs: Additional parameters (not used)

        Returns:
            tuple: (modified_df, train_metadata, test_metadata, val_metadata)
                - modified_df: DataFrame with 'split' column added, ordered by split files
                - train_metadata: Dictionary with train split information
                - test_metadata: Dictionary with test split information
                - val_metadata: Dictionary with validation split information
        """
        # Load the pre-existing splits (preserving order)
        train_df = pd.read_csv(self.train_split_path)
        test_df = pd.read_csv(self.test_split_path)
        val_df = pd.read_csv(self.val_split_path)

        # Extract the IDs (as ordered lists)
        train_ids_ordered = train_df[self.id_column].astype(str).tolist()
        test_ids_ordered = test_df[self.id_column].astype(str).tolist()
        val_ids_ordered = val_df[self.id_column].astype(str).tolist()

        # Convert to sets for overlap checking
        train_ids = set(train_ids_ordered)
        test_ids = set(test_ids_ordered)
        val_ids = set(val_ids_ordered)

        # Check for overlaps
        train_test_overlap = train_ids & test_ids
        train_val_overlap = train_ids & val_ids
        test_val_overlap = test_ids & val_ids

        if train_test_overlap:
            raise ValueError(f"Found {len(train_test_overlap)} IDs in both train and test splits: {train_test_overlap}")
        if train_val_overlap:
            raise ValueError(f"Found {len(train_val_overlap)} IDs in both train and val splits: {train_val_overlap}")
        if test_val_overlap:
            raise ValueError(f"Found {len(test_val_overlap)} IDs in both test and val splits: {test_val_overlap}")

        # Create ID to row mapping for fast lookup
        df_id_to_row = {}
        for idx, row in df.iterrows():
            row_id = str(row[self.id_column])
            df_id_to_row[row_id] = row

        # Build ordered dataframe following split file ordering
        ordered_rows = []
        train_ids_not_in_df = []
        test_ids_not_in_df = []
        val_ids_not_in_df = []

        # Add train rows in order
        for train_id in train_ids_ordered:
            if train_id in df_id_to_row:
                row = df_id_to_row[train_id].copy()
                row['split'] = 'train'
                ordered_rows.append(row)
            else:
                train_ids_not_in_df.append(train_id)

        # Add val rows in order
        for val_id in val_ids_ordered:
            if val_id in df_id_to_row:
                row = df_id_to_row[val_id].copy()
                row['split'] = 'val'
                ordered_rows.append(row)
            else:
                val_ids_not_in_df.append(val_id)

        # Add test rows in order
        for test_id in test_ids_ordered:
            if test_id in df_id_to_row:
                row = df_id_to_row[test_id].copy()
                row['split'] = 'test'
                ordered_rows.append(row)
            else:
                test_ids_not_in_df.append(test_id)

        # Check for IDs in df but not in any split file
        all_split_ids = train_ids | test_ids | val_ids
        df_ids = set(df[self.id_column].astype(str))
        ids_not_in_splits = df_ids - all_split_ids

        if train_ids_not_in_df:
            print(f"Warning: {len(train_ids_not_in_df)} train IDs not found in input dataframe")
        if test_ids_not_in_df:
            print(f"Warning: {len(test_ids_not_in_df)} test IDs not found in input dataframe")
        if val_ids_not_in_df:
            print(f"Warning: {len(val_ids_not_in_df)} val IDs not found in input dataframe")
        if ids_not_in_splits:
            print(f"Warning: {len(ids_not_in_splits)} IDs in dataframe but not in any split file (split will be None)")

        # Add rows that aren't in any split file (with split=None)
        for df_id in ids_not_in_splits:
            row = df_id_to_row[df_id].copy()
            row['split'] = None
            ordered_rows.append(row)

        # Create final dataframe from ordered rows
        result_df = pd.DataFrame(ordered_rows).reset_index(drop=True)

        # Summarize the splits
        train_count = (result_df['split'] == 'train').sum()
        val_count = (result_df['split'] == 'val').sum()
        test_count = (result_df['split'] == 'test').sum()

        print(f"Training examples: {train_count}")
        print(f"Validation examples: {val_count}")
        print(f"Testing examples: {test_count}")
        print(f"Total: {len(result_df)}")

        # Create metadata dictionaries
        train_metadata = {
            'source_file': str(self.train_split_path),
            'total_ids': len(train_ids),
            'ids_found_in_df': train_count,
            'ids_not_in_df': train_ids_not_in_df
        }

        test_metadata = {
            'source_file': str(self.test_split_path),
            'total_ids': len(test_ids),
            'ids_found_in_df': test_count,
            'ids_not_in_df': test_ids_not_in_df
        }

        val_metadata = {
            'source_file': str(self.val_split_path),
            'total_ids': len(val_ids),
            'ids_found_in_df': val_count,
            'ids_not_in_df': val_ids_not_in_df
        }

        return result_df, train_metadata, test_metadata, val_metadata
