import polars as pl


def transform_photos(df: pl.DataFrame) -> pl.DataFrame:
    # Convert to boolean
    df = df.with_columns(pl.col("photo_featured") == "t")
    return df


def transform_colors(df: pl.DataFrame) -> pl.DataFrame:
    # Convert to boolean
    df = df.cast({"red": pl.UInt8, "green": pl.UInt8, "blue": pl.UInt8})
    return df


def transform_keywords(df: pl.DataFrame) -> pl.DataFrame:
    dtypes = {
        "photo_id": pl.Categorical,
    }
    df = df.cast(dtypes)

    # Convert to boolean
    df = df.with_columns(pl.col("suggested_by_user") == "t")

    return df


def transform_collections(df: pl.DataFrame) -> pl.DataFrame:
    dtypes = {
        "photo_id": pl.Categorical,
        "collection_type": pl.Categorical,
    }
    df = df.cast(dtypes)

    # Transform datetimes
    df = df.with_columns(
        pl.col("photo_collected_at").str.to_datetime("%Y-%m-%d %H:%M:%S%.f")
    )

    return df


def transform_conversions(df: pl.DataFrame) -> pl.DataFrame:
    dtypes = {
        "conversion_type": pl.Categorical,
        "conversion_country": pl.Categorical,
        "photo_id": pl.Categorical,
    }
    df = df.cast(dtypes)

    # Transform datetimes
    df = df.with_columns(pl.col("converted_at").str.to_datetime("%Y-%m-%d %H:%M:%S%.f"))

    return df
