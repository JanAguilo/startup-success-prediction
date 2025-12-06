"""
Explainability Page with SHAP
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils import get_model_and_data

# Enhanced CSS for explainability page
st.markdown("""
    <style>
    .page-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
        text-align: center;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    .insight-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-header">🔍 Model Explainability with SHAP</h1>', unsafe_allow_html=True)
st.markdown("---")

# Show loading state
with st.spinner("Loading model and preparing SHAP explanations..."):
    try:
        model, df_raw, df_processed, X_train, X_test, y_train, y_test = get_model_and_data()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

try:
    
    # Get SHAP explainer
    @st.cache_resource
    def get_shap_explainer(_model):
        """Create and cache SHAP explainer"""
        explainer = shap.TreeExplainer(_model)
        return explainer
    
    # Sample data for faster computation
    sample_size = min(100, len(X_train))
    X_train_sample = X_train.sample(n=sample_size, random_state=42)
    
    explainer = get_shap_explainer(model)
    shap_values_raw = explainer.shap_values(X_train_sample)
    
    # Handle case where shap_values might be a list (for binary classification in newer SHAP versions)
    # According to notebook: newer versions return list, but we pass directly for plots
    if isinstance(shap_values_raw, list):
        # Check if it's a list of arrays - use the appropriate one
        if len(shap_values_raw) > 1:
            shap_values = shap_values_raw[1]  # Use positive class for binary classification
        else:
            shap_values = shap_values_raw[0]
    else:
        shap_values = shap_values_raw
    
    st.markdown('<h2 class="section-title">🌐 Global Feature Importance</h2>', unsafe_allow_html=True)
    st.markdown("Understanding which features are most important for the model's predictions overall.")
    
    # Summary plot (bar)
    st.markdown("#### Top Features by Mean |SHAP Value|")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    shap.summary_plot(shap_values, X_train_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.markdown("---")
    
    # Summary plot (dots)
    st.markdown("#### Feature Impact Summary")
    st.markdown("Each point represents a startup. Color indicates feature value (red=high, blue=low). Position shows impact on prediction.")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_train_sample, show=False, max_display=15)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.markdown("---")
    
    # Dependence plots for top features
    st.markdown('<h2 class="section-title">📉 Feature Dependence Plots</h2>', unsafe_allow_html=True)
    st.markdown("Explore how individual features interact with the model's predictions.")
    
    top_features = ['company_age', 'age_last_funding_year', 'relationships', 
                    'funding_total_usd', 'funding_duration', 'funding_rounds']
    top_features = [f for f in top_features if f in X_train_sample.columns]
    
    selected_feature = st.selectbox("Select Feature for Dependence Plot", 
                                   options=top_features, index=0)
    
    if selected_feature:
        # Create figure and axis for SHAP dependence plot
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # Plot SHAP dependence plot
        # Try with ax parameter first, fallback to without if it fails
        try:
            shap.dependence_plot(
                selected_feature, 
                shap_values, 
                X_train_sample, 
                ax=ax,
                show=False
            )
        except (TypeError, ValueError):
            # If ax parameter doesn't work, plot without it and capture the figure
            plt.close(fig)
            shap.dependence_plot(
                selected_feature, 
                shap_values, 
                X_train_sample, 
                show=False
            )
            fig = plt.gcf()
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
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
    
    # Individual prediction explanation - for predictions made on Predictions page
    st.markdown('<h2 class="section-title">🔬 Individual Prediction Explanation</h2>', unsafe_allow_html=True)
    st.markdown("View SHAP explanations for predictions made on the **Predictions** page.")
    
    # Check if there's a stored prediction
    if 'last_prediction' in st.session_state and st.session_state['last_prediction'].get('has_prediction', False):
        pred_data = st.session_state['last_prediction']
        instance = pred_data['input_df']
        prediction = pred_data['prediction']
        probability = pred_data['probability']
        
        # Calculate SHAP values for this instance
        instance_shap_raw = explainer.shap_values(instance)
        # Handle list format if needed
        if isinstance(instance_shap_raw, list):
            if len(instance_shap_raw) > 1:
                instance_shap = instance_shap_raw[1]
            else:
                instance_shap = instance_shap_raw[0]
        else:
            instance_shap = instance_shap_raw
        
        # Display prediction summary
        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == 1:
                st.success("✅ **Prediction: SUCCESS**")
            else:
                st.error("❌ **Prediction: FAILURE**")
        with col2:
            success_prob = probability[1] * 100
            st.metric("Success Probability", f"{success_prob:.2f}%")
        with col3:
            failure_prob = probability[0] * 100
            st.metric("Failure Probability", f"{failure_prob:.2f}%")
        
        st.info("💡 This explanation is for your most recent prediction from the **Predictions** page. Make a new prediction there to update this explanation.")
        
        # Waterfall plot for individual prediction
        st.markdown("#### SHAP Waterfall Plot")
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.waterfall_plot(
                shap.Explanation(
                    values=instance_shap[0],
                    base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                    data=instance.iloc[0].values,
                    feature_names=instance.columns.tolist()
                ),
                show=False,
                max_display=20
            )
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not display waterfall plot: {str(e)}")
            # Fallback: show force plot if available
            try:
                st.shap.waterfall_chart(
                    shap.Explanation(
                        values=instance_shap[0],
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                        data=instance.iloc[0].values,
                        feature_names=instance.columns.tolist()
                    )
                )
            except:
                st.error("Waterfall plot not available. Please check your SHAP installation.")
        
        # Feature contributions table
        st.markdown("#### Feature Contributions to Prediction")
        st.markdown("Features are ranked by their absolute SHAP value (impact on prediction).")
        
        # Get expected value (base value)
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        feature_df = pd.DataFrame({
            'Feature': instance.columns,
            'Feature Value': instance.iloc[0].values,
            'SHAP Value': instance_shap[0],
            'Contribution': instance_shap[0]  # Same as SHAP value
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        # Format the dataframe for better display
        feature_df['Impact'] = feature_df['SHAP Value'].apply(
            lambda x: '🔼 Increases Success' if x > 0 else '🔽 Decreases Success'
        )
        
        # Display top 20 features
        st.dataframe(
            feature_df[['Feature', 'Feature Value', 'SHAP Value', 'Impact']].head(20),
            use_container_width=True,
            hide_index=True
        )
        
        # Summary of top contributing features
        st.markdown("#### Top Contributing Features")
        top_positive = feature_df.nlargest(5, 'SHAP Value')
        top_negative = feature_df.nsmallest(5, 'SHAP Value')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Features Increasing Success Probability:**")
            for idx, row in top_positive.iterrows():
                st.write(f"• **{row['Feature']}**: {row['SHAP Value']:.4f} (value: {row['Feature Value']:.2f})")
        
        with col2:
            st.markdown("**Features Decreasing Success Probability:**")
            for idx, row in top_negative.iterrows():
                st.write(f"• **{row['Feature']}**: {row['SHAP Value']:.4f} (value: {row['Feature Value']:.2f})")
        
        st.markdown(f"**Base Value (Expected Prediction)**: {expected_value:.4f}")
        st.markdown(f"**Predicted Probability**: {probability[1]:.4f} = Base Value + Sum of SHAP Values")
    
    else:
        st.info("👆 **No prediction available yet!**\n\nTo see SHAP explanations for a specific prediction:\n1. Go to the **Predictions** page\n2. Fill in the startup characteristics\n3. Click 'Predict Success'\n4. Return here to see the explanation")
        
        # Optional: Show example from test set
        with st.expander("📊 View Example Explanation from Test Dataset"):
            sample_indices = X_test.index.tolist()[:50]
            selected_idx = st.selectbox("Select Example Startup Index", options=sample_indices, index=0, key="example_idx")
            
            if selected_idx is not None and selected_idx in X_test.index:
                example_instance = X_test.loc[[selected_idx]]
                example_actual = y_test.loc[selected_idx]
                example_prediction = model.predict(example_instance)[0]
                example_probability = model.predict_proba(example_instance)[0]
                
                example_shap_raw = explainer.shap_values(example_instance)
                if isinstance(example_shap_raw, list):
                    if len(example_shap_raw) > 1:
                        example_shap = example_shap_raw[1]
                    else:
                        example_shap = example_shap_raw[0]
                else:
                    example_shap = example_shap_raw
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Actual", "Success" if example_actual == 1 else "Failure")
                with col2:
                    st.metric("Predicted", "Success" if example_prediction == 1 else "Failure")
                with col3:
                    st.metric("Success Prob", f"{example_probability[1]*100:.2f}%")
                with col4:
                    match = "✅ Correct" if example_actual == example_prediction else "❌ Incorrect"
                    st.metric("Match", match)
                
                # Top features for example
                example_feature_df = pd.DataFrame({
                    'Feature': example_instance.columns,
                    'Value': example_instance.iloc[0].values,
                    'SHAP Value': example_shap[0]
                }).sort_values('SHAP Value', key=abs, ascending=False)
                
                st.dataframe(example_feature_df.head(15), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Model insights
    st.markdown('<h2 class="section-title">💡 Key Insights from SHAP Analysis</h2>', unsafe_allow_html=True)
    
    # Use markdown outside HTML div, then wrap in styled container
    st.markdown("""
    <div class="insight-box">
    <h3 style="color: #1e293b; margin-top: 0;">Top Contributing Features:</h3>
    <ol style="color: #1e293b; line-height: 1.8;">
        <li><strong>Company Age</strong>: The most important feature. Older companies have significantly higher success rates.</li>
        <li><strong>Age at Last Funding</strong>: Recent funding activity is a strong positive signal.</li>
        <li><strong>Relationships</strong>: Network effects matter - more connections correlate with success.</li>
        <li><strong>Funding Total</strong>: Capital raised is important, though with diminishing returns.</li>
        <li><strong>Funding Duration</strong>: Sustained investor interest indicates viability.</li>
    </ol>
    
    <h3 style="color: #1e293b; margin-top: 1.5rem;">Geographic and Industry Factors:</h3>
    <ul style="color: #1e293b; line-height: 1.8;">
        <li>Location (state) and industry category have smaller but meaningful impacts.</li>
        <li>California and New York show slight advantages.</li>
        <li>Software and web categories tend to perform well.</li>
    </ul>
    
    <h3 style="color: #1e293b; margin-top: 1.5rem;">Model Characteristics:</h3>
    <ul style="color: #1e293b; line-height: 1.8;">
        <li>The model captures complex non-linear relationships.</li>
        <li>Feature interactions are automatically learned by LightGBM.</li>
        <li>SHAP values provide consistent, additive explanations.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Error in explainability analysis: {str(e)}")
    st.exception(e)
    st.info("💡 **Troubleshooting:** Make sure the model is loaded correctly and SHAP is properly installed.")

