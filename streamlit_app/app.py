import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Startup Success Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Paths - Handle different execution contexts
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
# Get the project root (parent of streamlit_app)
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
    # Resolve any .. in the path
    try:
        resolved_path = path.resolve()
        if resolved_path.exists() and resolved_path.is_file():
            DATA_PATH = resolved_path
            break
    except (OSError, RuntimeError):
        continue

# If still not found, try searching from BASE_DIR
if DATA_PATH is None:
    data_dir = BASE_DIR / "data" / "raw"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            DATA_PATH = csv_files[0]

# Final fallback: try from current working directory
if DATA_PATH is None:
    cwd_data_path = Path(os.getcwd()) / "data" / "raw" / "startup_data.csv"
    if cwd_data_path.exists():
        DATA_PATH = cwd_data_path.resolve()

MODEL_PATH = SCRIPT_DIR / "model.pkl"
PREPROCESSOR_PATH = SCRIPT_DIR / "preprocessor.pkl"

# Helper functions
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

def years_between(d1, d2):
    """Calculate years between two dates"""
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    return (d2 - d1).days / 365.25

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
    """Train and cache the LightGBM model"""
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
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
    
    # Try to load saved model, otherwise train
    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        model = train_model(X_train, y_train)
        # Save model for future use
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    
    return model, df_raw, df_processed, X_train, X_test, y_train, y_test

# Sidebar navigation
st.sidebar.title("📈 Startup Success Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "📊 Exploratory Analysis", "🔮 Predictions", "🔍 Explainability"]
)

# Main content
if page == "🏠 Home":
    st.markdown('<div class="main-header">Startup Success Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Analysis, Predictions, and Explainability</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Exploratory Analysis")
        st.markdown("""
        - Feature distributions
        - Correlation heatmaps
        - Success trends (2000-2013)
        - Statistical insights
        """)
    
    with col2:
        st.markdown("### 🔮 Predictions")
        st.markdown("""
        - Input startup characteristics
        - Real-time success predictions
        - Probability scores
        - Model confidence
        """)
    
    with col3:
        st.markdown("### 🔍 Explainability")
        st.markdown("""
        - SHAP feature importance
        - Individual prediction explanations
        - Feature impact analysis
        - Model interpretability
        """)
    
    st.markdown("---")
    
    st.markdown("### About This Project")
    st.markdown("""
    This application provides an interactive platform for analyzing startup success factors, 
    making predictions, and understanding model decisions using SHAP (SHapley Additive exPlanations).
    
    **Key Features:**
    - **Data-driven insights**: Explore historical startup data from 2000-2013
    - **Machine learning predictions**: LightGBM model trained on 875 startups
    - **Explainable AI**: Understand which features drive each prediction
    - **Interactive visualizations**: Python-powered charts and heatmaps
    """)
    
    # Load and show dataset info
    try:
        df_raw = load_data()
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Startups", len(df_raw))
        with col2:
            st.metric("Features", len(df_raw.columns))
        with col3:
            acquired = (df_raw['status'] == 'acquired').sum()
            st.metric("Acquired", acquired)
        with col4:
            closed = (df_raw['status'] == 'closed').sum()
            st.metric("Closed", closed)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

elif page == "📊 Exploratory Analysis":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        df_raw, df_processed = preprocess_data(load_data())
        
        # Overview section
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Startups", len(df_raw))
        with col2:
            success_rate = df_processed['success'].mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            st.metric("Features", len(df_processed.columns) - 1)
        with col4:
            avg_funding = df_processed['funding_total_usd'].mean() / 1e6
            st.metric("Avg Funding (M)", f"${avg_funding:.2f}M")
        
        st.markdown("---")
        
        # Success trends over time
        st.markdown("### Success Trends Over Time (2000-2013)")
        df_raw['founded_at'] = pd.to_datetime(df_raw['founded_at'], errors='coerce')
        df_raw['founded_year'] = df_raw['founded_at'].dt.year
        
        yearly_stats = df_raw.groupby('founded_year').agg({
            'status': lambda x: (x == 'acquired').sum() / len(x) * 100
        }).reset_index()
        yearly_stats.columns = ['Year', 'Success Rate (%)']
        yearly_stats = yearly_stats.dropna()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(yearly_stats['Year'], yearly_stats['Success Rate (%)'], 
                marker='o', linewidth=2, markersize=8, color='#1f77b4')
        ax.set_xlabel('Year Founded', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Startup Success Rate by Founding Year', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(yearly_stats['Year'].astype(int))
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Feature distributions
        st.markdown("### Key Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Funding Total (Log Scale)")
            fig, ax = plt.subplots(figsize=(10, 6))
            log_funding = np.log1p(df_processed['funding_total_usd'])
            ax.hist(log_funding, bins=40, color='#2ca02c', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Log(Funding Total USD)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Total Funding (Log Scale)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Company Age Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_processed['company_age'].dropna(), bins=30, color='#ff7f0e', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Company Age (Years)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Company Age', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Funding Rounds")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_processed['funding_rounds'], bins=20, color='#d62728', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Funding Rounds', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Funding Rounds', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Relationships")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_processed['relationships'], bins=30, color='#9467bd', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Relationships', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Relationships', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Correlation heatmap
        st.markdown("### Correlation Heatmap - Key Numeric Features")
        
        numeric_cols = [
            'company_age', 'time_to_first_funding', 'funding_duration',
            'funding_rounds', 'funding_total_usd', 'avg_participants',
            'relationships', 'milestones', 'age_first_funding_year', 'age_last_funding_year'
        ]
        
        # Filter to existing columns
        numeric_cols = [col for col in numeric_cols if col in df_processed.columns]
        
        corr_matrix = df_processed[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Key Numeric Features', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Success by category
        st.markdown("### Success Rate by Category")
        
        category_cols = ['is_software', 'is_web', 'is_mobile', 'is_enterprise', 
                        'is_advertising', 'is_gamesvideo', 'is_ecommerce', 
                        'is_biotech', 'is_consulting']
        category_cols = [col for col in category_cols if col in df_processed.columns]
        
        category_success = []
        for col in category_cols:
            category_name = col.replace('is_', '').title()
            success_rate = df_processed[df_processed[col] == 1]['success'].mean() * 100
            category_success.append({'Category': category_name, 'Success Rate (%)': success_rate})
        
        category_df = pd.DataFrame(category_success).sort_values('Success Rate (%)', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(category_df['Category'], category_df['Success Rate (%)'], color='#1f77b4')
        ax.set_xlabel('Success Rate (%)', fontsize=12)
        ax.set_title('Success Rate by Industry Category', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Geographic analysis
        st.markdown("### Geographic Distribution")
        
        state_cols = ['is_CA', 'is_NY', 'is_MA', 'is_TX']
        state_cols = [col for col in state_cols if col in df_processed.columns]
        
        state_success = []
        state_names = {'is_CA': 'California', 'is_NY': 'New York', 
                      'is_MA': 'Massachusetts', 'is_TX': 'Texas'}
        
        for col in state_cols:
            state_name = state_names.get(col, col.replace('is_', ''))
            success_rate = df_processed[df_processed[col] == 1]['success'].mean() * 100
            state_success.append({'State': state_name, 'Success Rate (%)': success_rate})
        
        state_df = pd.DataFrame(state_success).sort_values('Success Rate (%)', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(state_df['State'], state_df['Success Rate (%)'], color='#2ca02c')
            ax.set_ylabel('Success Rate (%)', fontsize=12)
            ax.set_title('Success Rate by State', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.dataframe(state_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.exception(e)

elif page == "🔮 Predictions":
    st.markdown('<div class="main-header">🔮 Startup Success Prediction</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        model, df_raw, df_processed, X_train, X_test, y_train, y_test = get_model_and_data()
        
        st.markdown("### Enter Startup Characteristics")
        st.markdown("Fill in the details below to get a real-time prediction of startup success probability.")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Location & Geography")
                state_code = st.selectbox("State Code", 
                    options=sorted([col.replace('state_code_', '') for col in X_train.columns if col.startswith('state_code_')]),
                    index=0)
                
                is_CA = st.checkbox("California", value=False)
                is_NY = st.checkbox("New York", value=False)
                is_MA = st.checkbox("Massachusetts", value=False)
                is_TX = st.checkbox("Texas", value=False)
                is_otherstate = st.checkbox("Other State", value=False)
                
                latitude = st.number_input("Latitude", value=37.7749, format="%.4f")
                longitude = st.number_input("Longitude", value=-122.4194, format="%.4f")
            
            with col2:
                st.markdown("#### Funding Information")
                funding_total_usd = st.number_input("Total Funding (USD)", 
                    min_value=0, value=1000000, step=100000, format="%d")
                funding_rounds = st.number_input("Number of Funding Rounds", 
                    min_value=0, value=1, step=1)
                age_first_funding_year = st.number_input("Age at First Funding (Years)", 
                    min_value=0.0, value=2.0, step=0.1, format="%.2f")
                age_last_funding_year = st.number_input("Age at Last Funding (Years)", 
                    min_value=0.0, value=5.0, step=0.1, format="%.2f")
                avg_participants = st.number_input("Average Participants per Round", 
                    min_value=0.0, value=2.0, step=0.1, format="%.2f")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Company Lifecycle")
                company_age = st.number_input("Company Age (Years)", 
                    min_value=0.0, value=5.0, step=0.1, format="%.2f")
                time_to_first_funding = st.number_input("Time to First Funding (Years)", 
                    min_value=0.0, value=1.5, step=0.1, format="%.2f")
                funding_duration = st.number_input("Funding Duration (Years)", 
                    min_value=0.0, value=3.0, step=0.1, format="%.2f")
            
            with col2:
                st.markdown("#### Network & Milestones")
                relationships = st.number_input("Number of Relationships", 
                    min_value=0, value=5, step=1)
                milestones = st.number_input("Number of Milestones", 
                    min_value=0, value=1, step=1)
                has_milestones = st.checkbox("Has Milestones", value=True)
            
            st.markdown("#### Industry Category")
            col1, col2, col3 = st.columns(3)
            with col1:
                is_software = st.checkbox("Software", value=False)
                is_web = st.checkbox("Web", value=False)
                is_mobile = st.checkbox("Mobile", value=False)
            with col2:
                is_enterprise = st.checkbox("Enterprise", value=False)
                is_advertising = st.checkbox("Advertising", value=False)
                is_gamesvideo = st.checkbox("Games/Video", value=False)
            with col3:
                is_ecommerce = st.checkbox("E-commerce", value=False)
                is_biotech = st.checkbox("Biotech", value=False)
                is_consulting = st.checkbox("Consulting", value=False)
            
            st.markdown("#### Funding Type")
            col1, col2 = st.columns(2)
            with col1:
                has_VC = st.checkbox("Has VC Funding", value=False)
                has_angel = st.checkbox("Has Angel Funding", value=False)
                has_roundA = st.checkbox("Has Round A", value=False)
            with col2:
                has_roundB = st.checkbox("Has Round B", value=False)
                has_roundC = st.checkbox("Has Round C", value=False)
                has_roundD = st.checkbox("Has Round D", value=False)
            
            is_top500 = st.checkbox("Top 500 Company", value=False)
            
            submitted = st.form_submit_button("🔮 Predict Success", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {}
            
            # Add all features from training set
            for col in X_train.columns:
                if col.startswith('state_code_'):
                    input_data[col] = 1 if col == f'state_code_{state_code}' else 0
                else:
                    input_data[col] = 0
            
            # Update with user inputs
            input_data.update({
                'latitude': latitude,
                'longitude': longitude,
                'age_first_funding_year': age_first_funding_year,
                'age_last_funding_year': age_last_funding_year,
                'relationships': relationships,
                'funding_rounds': funding_rounds,
                'funding_total_usd': funding_total_usd,
                'milestones': milestones,
                'is_CA': 1 if is_CA else 0,
                'is_NY': 1 if is_NY else 0,
                'is_MA': 1 if is_MA else 0,
                'is_TX': 1 if is_TX else 0,
                'is_otherstate': 1 if is_otherstate else 0,
                'is_software': 1 if is_software else 0,
                'is_web': 1 if is_web else 0,
                'is_mobile': 1 if is_mobile else 0,
                'is_enterprise': 1 if is_enterprise else 0,
                'is_advertising': 1 if is_advertising else 0,
                'is_gamesvideo': 1 if is_gamesvideo else 0,
                'is_ecommerce': 1 if is_ecommerce else 0,
                'is_biotech': 1 if is_biotech else 0,
                'is_consulting': 1 if is_consulting else 0,
                'is_othercategory': 1 if not any([is_software, is_web, is_mobile, is_enterprise,
                                                  is_advertising, is_gamesvideo, is_ecommerce,
                                                  is_biotech, is_consulting]) else 0,
                'has_VC': 1 if has_VC else 0,
                'has_angel': 1 if has_angel else 0,
                'has_roundA': 1 if has_roundA else 0,
                'has_roundB': 1 if has_roundB else 0,
                'has_roundC': 1 if has_roundC else 0,
                'has_roundD': 1 if has_roundD else 0,
                'avg_participants': avg_participants,
                'is_top500': 1 if is_top500 else 0,
                'company_age': company_age,
                'time_to_first_funding': time_to_first_funding,
                'funding_duration': funding_duration,
                'has_milestones': 1 if has_milestones else 0,
            })
            
            # Create DataFrame with same column order as training data
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("✅ **Prediction: SUCCESS**")
                    st.balloons()
                else:
                    st.error("❌ **Prediction: FAILURE**")
            
            with col2:
                success_prob = probability[1] * 100
                st.metric("Success Probability", f"{success_prob:.2f}%")
            
            with col3:
                failure_prob = probability[0] * 100
                st.metric("Failure Probability", f"{failure_prob:.2f}%")
            
            # Probability bar
            st.markdown("#### Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 2))
            colors = ['#d62728', '#2ca02c']
            ax.barh(['Failure', 'Success'], [failure_prob, success_prob], color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability (%)', fontsize=11)
            ax.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
            for i, (label, prob) in enumerate(zip(['Failure', 'Success'], [failure_prob, success_prob])):
                ax.text(prob/2, i, f'{prob:.2f}%', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            plt.close()
            
            # Model performance info
            st.markdown("---")
            st.markdown("#### Model Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", len(X_train))
            with col2:
                st.metric("Test Accuracy", f"{(model.score(X_test, y_test)*100):.1f}%")
            with col3:
                from sklearn.metrics import roc_auc_score
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
                st.metric("ROC-AUC Score", f"{auc_score:.3f}")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.exception(e)

elif page == "🔍 Explainability":
    st.markdown('<div class="main-header">🔍 Model Explainability with SHAP</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        model, df_raw, df_processed, X_train, X_test, y_train, y_test = get_model_and_data()
        
        # Get SHAP explainer
        @st.cache_resource
        def get_shap_explainer(model, X_train_sample):
            """Create and cache SHAP explainer"""
            explainer = shap.TreeExplainer(model)
            return explainer
        
        # Sample data for faster computation
        sample_size = min(100, len(X_train))
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
        
        explainer = get_shap_explainer(model, X_train_sample)
        shap_values = explainer.shap_values(X_train_sample)
        
        st.markdown("### Global Feature Importance")
        st.markdown("Understanding which features are most important for the model's predictions overall.")
        
        # Summary plot (bar)
        st.markdown("#### Top Features by Mean |SHAP Value|")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_train_sample, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Summary plot (dots)
        st.markdown("#### Feature Impact Summary")
        st.markdown("Each point represents a startup. Color indicates feature value (red=high, blue=low). Position shows impact on prediction.")
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train_sample, show=False, max_display=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Dependence plots for top features
        st.markdown("### Feature Dependence Plots")
        st.markdown("Explore how individual features interact with the model's predictions.")
        
        top_features = ['company_age', 'age_last_funding_year', 'relationships', 
                        'funding_total_usd', 'funding_duration', 'funding_rounds']
        top_features = [f for f in top_features if f in X_train_sample.columns]
        
        selected_feature = st.selectbox("Select Feature for Dependence Plot", 
                                       options=top_features, index=0)
        
        if selected_feature:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(selected_feature, shap_values, X_train_sample, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown(f"**Interpretation for {selected_feature}:**")
            if selected_feature == 'company_age':
                st.info("Older companies generally have higher success probability. This makes sense as they've had more time to reach acquisition.")
            elif selected_feature == 'age_last_funding_year':
                st.info("Recent funding activity (lower age at last funding) is a strong positive signal for success.")
            elif selected_feature == 'relationships':
                st.info("More relationships in the ecosystem correlate with higher success rates, indicating network effects.")
            elif selected_feature == 'funding_total_usd':
                st.info("Higher total funding increases success probability, though the relationship may be non-linear.")
            elif selected_feature == 'funding_duration':
                st.info("Longer funding duration suggests sustained investor interest and operational longevity.")
            elif selected_feature == 'funding_rounds':
                st.info("More funding rounds indicate growth trajectory and investor confidence.")
        
        st.markdown("---")
        
        # Individual prediction explanation
        st.markdown("### Individual Prediction Explanation")
        st.markdown("Select a startup from the dataset to see how each feature contributed to its prediction.")
        
        sample_indices = X_test.index.tolist()[:50]  # Show first 50 for selection
        selected_idx = st.selectbox("Select Startup Index", options=sample_indices, index=0)
        
        if selected_idx is not None and selected_idx in X_test.index:
            # Get prediction for selected instance
            instance = X_test.loc[[selected_idx]]
            actual = y_test.loc[selected_idx]
            prediction = model.predict(instance)[0]
            probability = model.predict_proba(instance)[0]
            
            # Calculate SHAP values for this instance
            instance_shap = explainer.shap_values(instance)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Actual", "Success" if actual == 1 else "Failure")
            with col2:
                st.metric("Predicted", "Success" if prediction == 1 else "Failure")
            with col3:
                st.metric("Success Prob", f"{probability[1]*100:.2f}%")
            with col4:
                match = "✅ Correct" if actual == prediction else "❌ Incorrect"
                st.metric("Match", match)
            
            # Waterfall plot for individual prediction
            st.markdown("#### SHAP Waterfall Plot")
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=instance_shap[0],
                    base_values=explainer.expected_value,
                    data=instance.iloc[0].values,
                    feature_names=instance.columns.tolist()
                ),
                show=False,
                max_display=20
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Feature values for this instance
            st.markdown("#### Feature Values for Selected Startup")
            feature_df = pd.DataFrame({
                'Feature': instance.columns,
                'Value': instance.iloc[0].values,
                'SHAP Value': instance_shap[0]
            }).sort_values('SHAP Value', key=abs, ascending=False)
            
            st.dataframe(feature_df.head(20), use_container_width=True)
        
        st.markdown("---")
        
        # Model insights
        st.markdown("### Key Insights from SHAP Analysis")
        st.markdown("""
        **Top Contributing Features:**
        1. **Company Age**: The most important feature. Older companies have significantly higher success rates.
        2. **Age at Last Funding**: Recent funding activity is a strong positive signal.
        3. **Relationships**: Network effects matter - more connections correlate with success.
        4. **Funding Total**: Capital raised is important, though with diminishing returns.
        5. **Funding Duration**: Sustained investor interest indicates viability.
        
        **Geographic and Industry Factors:**
        - Location (state) and industry category have smaller but meaningful impacts.
        - California and New York show slight advantages.
        - Software and web categories tend to perform well.
        
        **Model Characteristics:**
        - The model captures complex non-linear relationships.
        - Feature interactions are automatically learned by LightGBM.
        - SHAP values provide consistent, additive explanations.
        """)
    
    except Exception as e:
        st.error(f"Error in explainability analysis: {str(e)}")
        st.exception(e)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Authors:** Jan Aguiló and Pau Chaves")
st.sidebar.markdown("**Course:** Visual Analytics - Final Project")
st.sidebar.markdown("**Model:** LightGBM Classifier")

