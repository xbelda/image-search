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
