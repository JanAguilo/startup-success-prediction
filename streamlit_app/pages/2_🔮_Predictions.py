"""
Predictions Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_model_and_data

# Enhanced CSS for predictions page
st.markdown("""
    <style>
    .page-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea !important;
        margin-bottom: 1rem;
        text-align: center;
    }
    .prediction-result-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        color: #1e293b !important;
    }
    
    .prediction-result-card h2 {
        color: #1e293b !important;
        margin-top: 0;
    }
    
    .page-header {
        color: #667eea !important;
        -webkit-text-fill-color: #667eea !important;
    }
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #10b981;
    }
    .failure-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ef4444;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-header">🔮 Startup Success Prediction</h1>', unsafe_allow_html=True)
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
            state_options = sorted([col.replace('state_code_', '') for col in X_train.columns if col.startswith('state_code_')])
            ca_index = state_options.index('CA') if 'CA' in state_options else 0
            state_code = st.selectbox("State Code", 
                options=state_options,
                index=ca_index)
            
            is_CA = st.checkbox("California", value=True)
            is_NY = st.checkbox("New York", value=False)
            is_MA = st.checkbox("Massachusetts", value=False)
            is_TX = st.checkbox("Texas", value=False)
            is_otherstate = st.checkbox("Other State", value=False)
        
        with col2:
            st.markdown("#### Funding Information")
            funding_total_usd = st.number_input("Total Funding (USD)", 
                min_value=0, value=4200000, step=100000, format="%d")
            funding_rounds = st.number_input("Number of Funding Rounds", 
                min_value=0, value=2, step=1)
            age_first_funding_year = st.number_input("Age at First Funding (Years)", 
                min_value=0.0, value=2.0, step=0.1, format="%.2f")
            age_last_funding_year = st.number_input("Age at Last Funding (Years)", 
                min_value=0.0, value=4.0, step=0.1, format="%.2f")
            avg_participants = st.number_input("Average Participants per Round", 
                min_value=0.0, value=4.0, step=0.1, format="%.2f")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Company Lifecycle")
            company_age = st.number_input("Company Age (Years)", 
                min_value=0.0, value=5.0, step=0.1, format="%.2f")
            time_to_first_funding = st.number_input("Time to First Funding (Years)", 
                min_value=0.0, value=2.0, step=0.1, format="%.2f")
            funding_duration = st.number_input("Funding Duration (Years)", 
                min_value=0.0, value=2.0, step=0.1, format="%.2f")
        
        with col2:
            st.markdown("#### Network & Milestones")
            relationships = st.number_input("Number of Relationships", 
                min_value=0, value=8, step=1)
            milestones = st.number_input("Number of Milestones", 
                min_value=0, value=5, step=1)
            has_milestones = st.checkbox("Has Milestones", value=True)
        
        st.markdown("#### Industry Category")
        col1, col2, col3 = st.columns(3)
        with col1:
            is_software = st.checkbox("Software", value=True)
            is_web = st.checkbox("Web", value=True)
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
            has_VC = st.checkbox("Has VC Funding", value=True)
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
        # Set default latitude/longitude (model requires these but user doesn't input them)
        default_latitude = 37.7749  # Default to San Francisco area
        default_longitude = -122.4194
        
        input_data.update({
            'latitude': default_latitude,
            'longitude': default_longitude,
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
        
        # Store prediction data in session state for explainability page
        st.session_state['last_prediction'] = {
            'input_df': input_df,
            'prediction': prediction,
            'probability': probability,
            'has_prediction': True
        }
        
        st.markdown("---")
        st.markdown("""
        <div class="prediction-result-card">
            <h2 style="margin-top:0; color: #1e293b !important;">🎯 Prediction Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="success-box">
                    <h2 style="color: #059669; margin:0;">✅ Prediction: SUCCESS</h2>
                    <p style="color: #047857; margin:0.5rem 0 0 0;">Your startup is predicted to succeed!</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("""
                <div class="failure-box">
                    <h2 style="color: #dc2626; margin:0;">❌ Prediction: FAILURE</h2>
                    <p style="color: #991b1b; margin:0.5rem 0 0 0;">Your startup is predicted to fail.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            success_prob = probability[1] * 100
            st.metric("Success Probability", f"{success_prob:.2f}%", delta=None, delta_color="normal")
        
        with col3:
            failure_prob = probability[0] * 100
            st.metric("Failure Probability", f"{failure_prob:.2f}%", delta=None, delta_color="normal")
        
        # Information about explainability
        st.markdown("---")
        st.markdown("### Understand Your Prediction")
        st.info("💡 **Want to know which features influenced this prediction?** Visit the **🔍 Explainability** page to see a detailed SHAP explanation of this prediction.")
        
        # Probability bar
        st.markdown("---")
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

