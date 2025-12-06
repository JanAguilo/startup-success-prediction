"""
Shared utilities for the Streamlit app
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Paths - Handle different execution contexts
SCRIPT_DIR = Path(__file__).parent.absolute()
BASE_DIR = SCRIPT_DIR.parent.absolute()

# Try multiple possible paths for the data file
possible_data_paths = [
    BASE_DIR / "data" / "raw" / "startup_data.csv",
    SCRIPT_DIR / ".." / "data" / "raw" / "startup_data.csv",
    Path("data/raw/startup_data.csv").absolute(),
    Path("../data/raw/startup_data.csv").absolute(),
]

DATA_PATH = None
for path in possible_data_paths:
    try:
        resolved_path = path.resolve()
        if resolved_path.exists() and resolved_path.is_file():
            DATA_PATH = resolved_path
            break
    except (OSError, RuntimeError):
        continue

if DATA_PATH is None:
    data_dir = BASE_DIR / "data" / "raw"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            DATA_PATH = csv_files[0]

if DATA_PATH is None:
    cwd_data_path = Path(os.getcwd()) / "data" / "raw" / "startup_data.csv"
    if cwd_data_path.exists():
        DATA_PATH = cwd_data_path.resolve()

MODEL_PATH_NOTEBOOK = BASE_DIR / "models" / "model.pkl"
MODEL_PATH_LOCAL = SCRIPT_DIR / "model.pkl"

def years_between(d1, d2):
    """Calculate years between two dates"""
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    return (d2 - d1).days / 365.25

@st.cache_data
def load_data():
    """Load and cache the startup dataset"""
    if DATA_PATH is None or not DATA_PATH.exists():
        error_msg = f"❌ Data file 'startup_data.csv' not found!\n\n"
        error_msg += f"**Searched locations:**\n"
        for i, path in enumerate(possible_data_paths, 1):
            try:
                resolved = path.resolve()
                exists = "✓" if resolved.exists() else "✗"
                error_msg += f"{i}. {exists} {resolved}\n"
            except:
                error_msg += f"{i}. ✗ {path}\n"
        error_msg += f"\n**Current working directory:** {os.getcwd()}\n"
        error_msg += f"**Script directory:** {SCRIPT_DIR}\n"
        error_msg += f"**Base directory:** {BASE_DIR}\n\n"
        error_msg += f"Please ensure 'startup_data.csv' exists in 'data/raw/' directory."
        st.error(error_msg)
        raise FileNotFoundError(f"Data file not found at any expected location")
    
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess data following notebook pipeline"""
    df = df.copy()
    
    # Create target variable
    df['success'] = df['status'].map({'acquired': 1, 'closed': 0})
    
    # Drop non-informative columns
    cols_to_drop = [
        'Unnamed: 0', 'Unnamed: 6', 'id', 'object_id',
        'state_code.1', 'labels'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Convert date columns
    date_cols = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Feature engineering
    df["company_age"] = df.apply(
        lambda row: years_between(
            row["founded_at"],
            row["closed_at"] if pd.notna(row["closed_at"]) else row["last_funding_at"]
        ),
        axis=1
    )
    
    df["time_to_first_funding"] = df.apply(
        lambda row: years_between(row["founded_at"], row["first_funding_at"]),
        axis=1
    )
    
    df["funding_duration"] = df.apply(
        lambda row: years_between(row["first_funding_at"], row["last_funding_at"]),
        axis=1
    )
    
    df["has_milestones"] = (df["milestones"] > 0).astype(int)
    
    # Impute missing values
    df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(0)
    df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(0)
    
    # Remove invalid rows
    df = df[(df['company_age'] >= 0) & (df['time_to_first_funding'] >= 0)]
    
    # Drop columns for modeling
    drop_cols = [
        "name", "city", "zip_code", "category_code", "status",
        "founded_at", "first_funding_at", "closed_at", "last_funding_at",
        "age_first_milestone_year", "age_last_milestone_year",
    ]
    
    df_model = df.drop(columns=drop_cols, errors='ignore')
    
    # Encode categorical variables
    df_model_encoded = pd.get_dummies(df_model, columns=["state_code"], drop_first=True)
    
    return df, df_model_encoded

@st.cache_resource
def train_model(X_train, y_train):
    """Train and cache the LightGBM model with best hyperparameters"""
    model = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.7,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=10,
        random_state=42,
        class_weight="balanced",
        verbosity=-1
    )
    model.fit(X_train, y_train)
    return model

def get_model_and_data():
    """Get or train model and return with data"""
    df_raw, df_processed = preprocess_data(load_data())
    
    X = df_processed.drop(columns=["success"])
    y = df_processed["success"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Try to load saved model from models folder (trained in notebook) first
    model = None
    if MODEL_PATH_NOTEBOOK.exists():
        with open(MODEL_PATH_NOTEBOOK, 'rb') as f:
            model = pickle.load(f)
    elif MODEL_PATH_LOCAL.exists():
        with open(MODEL_PATH_LOCAL, 'rb') as f:
            model = pickle.load(f)
    else:
        # Train new model if no saved model exists
        model = train_model(X_train, y_train)
        # Save model for future use in local folder
        with open(MODEL_PATH_LOCAL, 'wb') as f:
            pickle.dump(model, f)
    
    return model, df_raw, df_processed, X_train, X_test, y_train, y_test

