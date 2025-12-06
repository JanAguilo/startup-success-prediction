"""
Main Streamlit app entry point
"""
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Welcome! - Startup Success Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 1.4rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 2.5rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1), 0 6px 6px rgba(0, 0, 0, 0.08);
    }
    
    .feature-card h3 {
        color: #1e293b;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    .feature-card ul {
        color: #475569;
        line-height: 1.8;
        margin: 0;
        padding-left: 1.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        color: white;
        text-align: center;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* About Section */
    .about-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
        color: #1e293b !important;
    }
    
    .about-section h2 {
        color: #1e293b !important;
    }
    
    /* Styled List */
    .styled-list {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        color: #1e293b !important;
    }
    
    .styled-list p {
        color: #1e293b !important;
    }
    
    .about-section {
        color: #1e293b !important;
    }
    
    .about-section p {
        color: #1e293b !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Divider Styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: white; font-size: 1.8rem; margin-bottom: 0.5rem;">🚀 Startup Success</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">Prediction Platform</p>
    </div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px; margin: 1rem 0;">
        <p style="color: white; margin: 0.5rem 0;"><strong>👨‍💻 Authors:</strong><br/>Jan Aguiló & Pau Chaves</p>
        <p style="color: white; margin: 0.5rem 0;"><strong>📚 Course:</strong><br/>Visual Analytics</p>
        <p style="color: white; margin: 0.5rem 0;"><strong>🤖 Model:</strong><br/>LightGBM Classifier</p>
    </div>
""", unsafe_allow_html=True)

# Home page content
st.markdown('<div class="main-header">Welcome!</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Startup Success Prediction & Analysis Platform</div>', unsafe_allow_html=True)

st.markdown("---")

# Feature Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Exploratory Analysis</h3>
        <ul>
            <li>Feature distributions</li>
            <li>Correlation heatmaps</li>
            <li>Success trends (2000-2013)</li>
            <li><strong>Dynamic filtering</strong></li>
            <li>Statistical insights</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔮 Predictions</h3>
        <ul>
            <li>Input startup characteristics</li>
            <li>Real-time success predictions</li>
            <li>Probability scores</li>
            <li>Model confidence metrics</li>
            <li>Interactive predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Explainability</h3>
        <ul>
            <li>SHAP feature importance</li>
            <li>Individual prediction explanations</li>
            <li>Feature impact analysis</li>
            <li>Model interpretability</li>
            <li>Interactive visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# About Section
st.markdown("""
    <div class="about-section">
        <h2 class="section-header">About This Project</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: #475569; margin-bottom: 1.5rem;">
            This application provides an interactive platform for analyzing startup success factors, 
            making predictions, and understanding model decisions using <strong>SHAP</strong> (SHapley Additive exPlanations).
        </p>
        <div class="styled-list">
            <p style="margin: 0.5rem 0; color: #1e293b !important;"><strong style="color: #1e293b;">📈 Data-driven insights:</strong> Explore historical startup data from 2000-2013</p>
            <p style="margin: 0.5rem 0; color: #1e293b !important;"><strong style="color: #1e293b;">🤖 Machine learning predictions:</strong> LightGBM model trained on 875 startups</p>
            <p style="margin: 0.5rem 0; color: #1e293b !important;"><strong style="color: #1e293b;">🧠 Explainable AI:</strong> Understand which features drive each prediction</p>
            <p style="margin: 0.5rem 0; color: #1e293b !important;"><strong style="color: #1e293b;">📊 Interactive visualizations:</strong> Python-powered charts and heatmaps</p>
            <p style="margin: 0.5rem 0; color: #1e293b !important;"><strong style="color: #1e293b;">🔍 Dynamic filtering:</strong> Filter data by multiple criteria in exploratory analysis</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Dataset Overview with enhanced metrics
try:
    from utils import load_data
    df_raw = load_data()
    st.markdown("""
        <h2 class="section-header">Dataset Overview</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Startups", f"{len(df_raw):,}", help="Total number of startups in the dataset")
    with col2:
        st.metric("📋 Features", len(df_raw.columns), help="Number of features available for analysis")
    with col3:
        acquired = (df_raw['status'] == 'acquired').sum()
        st.metric("✅ Acquired", f"{acquired:,}", help="Number of successfully acquired startups")
    with col4:
        closed = (df_raw['status'] == 'closed').sum()
        st.metric("❌ Closed", f"{closed:,}", help="Number of closed/failed startups")
        
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
