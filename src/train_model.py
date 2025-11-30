from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

from . import config
from .preprocessing import (
    load_clean_data,
    load_raw_data,
    clean_companies,
    get_feature_matrix,
    save_clean_data,
)


def ensure_clean_data() -> pd.DataFrame:
    """
    Load cleaned data if it exists, otherwise create it from raw.
    """
    path = config.DATA_PROCESSED_PATH
    if Path(path).exists():
        print(f"Loading cleaned data from {path}")
        return load_clean_data(path)

    print("Cleaned data not found. Generating from raw CSV...")
    df_raw = load_raw_data(config.DATA_RAW_PATH)
    df_clean = clean_companies(df_raw)
    save_clean_data(df_clean, path)
    print(f"Saved cleaned data to {path}")
    return df_clean


def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    """
    Build a full sklearn Pipeline: preprocessing + LightGBM classifier.
    """
    # Numeric: impute missing with median
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical: impute missing with most_freq + one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = LGBMClassifier(
        random_state=config.RANDOM_STATE,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf


def train_and_evaluate() -> Pipeline:
    """
    Main training loop: load data, split, train, evaluate, save model.
    """
    df_clean = ensure_clean_data()
    print("Clean data shape:", df_clean.shape)

    X, y, numeric_cols, categorical_cols = get_feature_matrix(df_clean)
    print(f"Using {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    clf = build_pipeline(numeric_cols, categorical_cols)

    print("Fitting model...")
    clf.fit(X_train, y_train)

    # --- Evaluation ---
    print("\n=== Evaluation on holdout set ===")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, digits=3))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f}")

    # --- Save model ---
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, config.MODEL_PATH)
    print(f"\nSaved trained pipeline to: {config.MODEL_PATH}")

    return clf


if __name__ == "__main__":
    train_and_evaluate()
