from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class CorpusSplitter(ABC):
    """
    Base class for splitting corpus data into train/validation/test sets.
    """

    @abstractmethod
    def add_split_column(self, df: pd.DataFrame, random_state: Any, **kwargs):
        """
        Add train/val/test split column to the dataframe.

        Args:
            df: DataFrame to add splits to
            random_state: Random state for reproducibility
            **kwargs: Additional parameters specific to the implementation

        Returns:
            tuple: (modified_df, train_metadata, test_metadata, val_metadata)
        """
        pass
