from pathlib import Path

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "companies.csv"
DATA_PROCESSED_PATH = BASE_DIR / "data" / "processed" / "companies_clean.parquet"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "startup_success_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------- LABEL DEFINITION ----------
# We only keep companies that are clearly successful or failed.
STATUS_SUCCESS = ["acquired", "ipo"]
STATUS_FAILURE = ["closed"]
TARGET_COL = "success"  # 1 = success, 0 = failure

# ---------- COLUMNS ----------
# Original date columns from Crunchbase dump
RAW_DATE_COLS = [
    "founded_at",
    "first_funding_at",
    "last_funding_at",
    "first_milestone_at",
    "last_milestone_at",
    "created_at",
    "updated_at",
]

# Numeric columns we will use directly (after imputation)
NUMERIC_BASE_COLS = [
    "funding_rounds",
    "funding_total_usd",
    "investment_rounds",
    "invested_companies",
    "milestones",
    "relationships",
    "logo_width",
    "logo_height",
    "lat",
    "lng",
]

# New numeric features we will engineer from dates
ENGINEERED_NUMERIC_COLS = [
    "founded_year",
    "founded_month",
    "age_at_first_funding_days",
    "age_at_last_funding_days",
    "age_at_first_milestone_days",
    "age_at_last_milestone_days",
    "company_age_at_last_update_days",
    "log_funding_total_usd",
    "has_twitter",
    "has_homepage",
]

NUMERIC_FEATURES = NUMERIC_BASE_COLS + ENGINEERED_NUMERIC_COLS

# Categorical columns (moderate cardinality – good for one-hot encoding)
CATEGORICAL_FEATURES = [
    "category_code",
    "country_code",
    "state_code",
    "region",
]

# Columns we will keep in the cleaned dataset (for debugging / EDA)
KEEP_COLS_BASE = [
    "id",
    "name",
    "status",
    TARGET_COL,
]

KEEP_COLS = (
    KEEP_COLS_BASE
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)
