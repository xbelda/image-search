from typing import Tuple

import pandas as pd


class KeywordProcessor:
    def __init__(self, min_num_counts: int = 100, padding_size: int = 10):
        self.min_num_counts = min_num_counts
        self.padding_size = padding_size

    def filter_out_uncommon_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out uncommon keywords that appear less than min_num_counts times
        """
        keyword_counts = df["keyword"].value_counts()
        valid_keywords = keyword_counts[keyword_counts >= self.min_num_counts].index
        df = df[df["keyword"].isin(valid_keywords)]
        return df.copy()

    def add_keyword_ids(self, df):
        keyword_ids, keyword_names = pd.factorize(df["keyword"])
        df["keyword_id"] = keyword_ids + self.padding_size
        return df

    def process(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = self.filter_out_uncommon_keywords(df)
        df = self.add_keyword_ids(df)
        df = df[["photo_id", "keyword", "keyword_id"]]
        return df


def temporal_train_test_split(
    conversions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporally splits the dataset in Train/Test
    This is one of the best approaches, since it allows us to measure more how the model would work under a more
    "realistic" scenario. That is, training a model on previous data and seeing how it evolves in the future.

    Args:
        conversions:

    Returns:

    """
    num_examples = len(conversions)
    conversions_train = conversions[: int(num_examples * 0.8)]
    conversions_val = conversions[int(num_examples * 0.8) :]
    return conversions_train, conversions_val


def load_and_preprocess_data() -> pd.DataFrame:
    """
    Reads the conversion dataset and sorts it temporally.
    """
    conversions = pd.read_csv(
        "./data/unsplash-research-dataset-lite-latest/conversions.tsv000",
        sep="\t",
        header=0,
    )
    conversions["converted_at"] = pd.to_datetime(
        conversions["converted_at"], format="ISO8601"
    )
    # Sort by conversion datetime
    conversions = conversions.sort_values("converted_at", ignore_index=True)
    return conversions
