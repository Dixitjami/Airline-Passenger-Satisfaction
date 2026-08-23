"""
Data ingestion component for the Airline Passenger Satisfaction project.

Loads ``dataset/train.csv`` and ``dataset/test.csv`` with logging and
custom exception handling.
"""

import os
import sys
import logging
import pandas as pd

try:  # support both "python src/... " script runs and package imports
    from exception import CustomException
    from logger import LOG_FILE_PATH  # noqa: F401  (configures logging)
except ImportError:  # pragma: no cover
    from ..exception import CustomException
    from ..logger import LOG_FILE_PATH  # noqa: F401  (configures logging)


class DataIngestion:
    """
    Loads the raw Airline Passenger Satisfaction train/test datasets.

    Parameters
    ----------
    train_path : str
        Path to the training CSV.
    test_path : str
        Path to the test CSV.
    """

    def __init__(self, train_path=None, test_path=None):
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        dataset_dir = os.path.join(project_root, "dataset")

        self.train_path = train_path or os.path.join(dataset_dir, "train.csv")
        self.test_path = test_path or os.path.join(dataset_dir, "test.csv")

    def _load_csv(self, path: str, label: str) -> pd.DataFrame:
        """Load a single CSV file with logging and error handling."""
        logging.info("Loading %s dataset from [%s]", label, path)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logging.error("Failed to load %s dataset from [%s]: %s", label, path, e)
            raise CustomException(e, sys)

        logging.info("%s dataset loaded successfully", label.capitalize())
        logging.info("%s dataset shape: %s", label.capitalize(), df.shape)
        return df

    def load_raw_data(self):
        """
        Load both training and test datasets.

        Returns
        -------
        tuple of pd.DataFrame
            (train_df, test_df)
        """
        logging.info("Starting Airline Passenger Satisfaction data ingestion")

        try:
            if not os.path.exists(self.train_path):
                raise FileNotFoundError(f"Training file not found: {self.train_path}")
            if not os.path.exists(self.test_path):
                raise FileNotFoundError(f"Test file not found: {self.test_path}")

            train_df = self._load_csv(self.train_path, "training")
            test_df = self._load_csv(self.test_path, "test")
        except CustomException:
            raise
        except Exception as e:
            logging.error("Data ingestion failed: %s", e)
            raise CustomException(e, sys)

        logging.info("Airline Passenger Satisfaction data ingestion completed")
        return train_df, test_df


def load_raw_data(train_path=None, test_path=None):
    """
    Convenience function matching the existing ``preprocessing.load_data``
    interface. Loads and returns (train_df, test_df).
    """
    ingestion = DataIngestion(train_path=train_path, test_path=test_path)
    return ingestion.load_raw_data()
