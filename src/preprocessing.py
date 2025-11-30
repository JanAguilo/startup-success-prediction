import numpy as np
import pandas as pd
from pathlib import Path

from . import config


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the raw Crunchbase companies CSV.
    """
    if path is None:
        path = config.DATA_RAW_PATH

    df = pd.read_csv(path)
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns to datetime.
    """
    df = df.copy()
    for col in config.RAW_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features (ages, durations, etc.).
    All in DAYS so that LightGBM can work easily.
    """
    df = df.copy()

    # founded year / month
    df["founded_year"] = df["founded_at"].dt.year
    df["founded_month"] = df["founded_at"].dt.month

    # Helper for safe timedelta in days
    def days_between(later, earlier):
        return (later - earlier).dt.days

    # Age at first / last funding
    df["age_at_first_funding_days"] = days_between(df["first_funding_at"], df["founded_at"])
    df["age_at_last_funding_days"] = days_between(df["last_funding_at"], df["founded_at"])

    # Age at first / last milestone
    df["age_at_first_milestone_days"] = days_between(
        df["first_milestone_at"], df["founded_at"]
    )
    df["age_at_last_milestone_days"] = days_between(
        df["last_milestone_at"], df["founded_at"]
    )

    # Company age at last update
    df["company_age_at_last_update_days"] = days_between(df["updated_at"], df["founded_at"])

    # Log funding (to reduce skew)
    df["log_funding_total_usd"] = np.log1p(df["funding_total_usd"])

    # Binary indicators
    df["has_twitter"] = df["twitter_username"].notna().astype(int)
    df["has_homepage"] = df["homepage_url"].notna().astype(int)

    return df


def _define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to companies with clear outcome, and define TARGET_COL.
    success = acquired or ipo
    failure = closed
    """
    df = df.copy()

    df["status"] = df["status"].str.lower()

    allowed_status = set(config.STATUS_SUCCESS + config.STATUS_FAILURE)
    df = df[df["status"].isin(allowed_status)]

    df[config.TARGET_COL] = df["status"].map(
        {**{s: 1 for s in config.STATUS_SUCCESS}, **{s: 0 for s in config.STATUS_FAILURE}}
    )

    return df


def clean_companies(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the companies dataset:
    - keep only entity_type='Company'
    - parse dates
    - engineer features
    - define success label
    - select useful columns
    """
    df = df_raw.copy()

    # 1) Filter only Company entities
    if "entity_type" in df.columns:
        df = df[df["entity_type"] == "Company"]

    # 2) Parse date columns
    df = _parse_dates(df)

    # 3) Define target from status
    df = _define_target(df)

    # 4) Engineer additional numeric features
    df = _engineer_time_features(df)

    # 5) Keep only relevant columns (for modeling + debugging)
    cols_to_keep = [c for c in config.KEEP_COLS if c in df.columns]
    df_clean = df[cols_to_keep].copy()

    return df_clean


def get_feature_matrix(df_clean: pd.DataFrame):
    """
    Given a cleaned dataframe, return X, y and lists of numeric / categorical
    feature names (for ColumnTransformer).
    """
    y = df_clean[config.TARGET_COL].astype(int)

    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in df_clean.columns]
    categorical_cols = [c for c in config.CATEGORICAL_FEATURES if c in df_clean.columns]

    X = df_clean[numeric_cols + categorical_cols].copy()

    return X, y, numeric_cols, categorical_cols


def save_clean_data(df_clean: pd.DataFrame, path: Path | str | None = None) -> None:
    """
    Save cleaned data to Parquet for faster downstream loading.
    """
    if path is None:
        path = config.DATA_PROCESSED_PATH

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(path, index=False)


def load_clean_data(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load cleaned data (if you've already run preprocessing once).
    """
    if path is None:
        path = config.DATA_PROCESSED_PATH
    return pd.read_parquet(path)


if __name__ == "__main__":
    # Small CLI-style entrypoint so you can run:
    #   python -m src.preprocessing
    print("Loading raw data from:", config.DATA_RAW_PATH)
    df_raw = load_raw_data()

    print("Cleaning...")
    df_clean = clean_companies(df_raw)
    print(f"Clean dataset shape: {df_clean.shape}")

    print("Saving cleaned data to:", config.DATA_PROCESSED_PATH)
    save_clean_data(df_clean)
    print("Done.")
