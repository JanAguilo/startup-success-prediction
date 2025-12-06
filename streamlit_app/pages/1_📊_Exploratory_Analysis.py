"""
Exploratory Data Analysis Page with Dynamic Filtering
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from utils import load_data, preprocess_data

# Enhanced CSS for this page
st.markdown("""
    <style>
    .page-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

try:
    df_raw, df_processed = preprocess_data(load_data())
    
    # Add founded_year for filtering
    df_raw['founded_at'] = pd.to_datetime(df_raw['founded_at'], errors='coerce')
    df_raw['founded_year'] = df_raw['founded_at'].dt.year
    df_processed_with_year = df_processed.copy()
    df_processed_with_year['founded_year'] = df_raw.loc[df_processed.index, 'founded_year'].values
    
    # ==================== FILTERS SIDEBAR ====================
    st.sidebar.markdown("### 🔍 Filters")
    st.sidebar.markdown("Apply filters to explore the data:")
    
    # Success status filter
    success_filter = st.sidebar.multiselect(
        "Success Status",
        options=["Success (Acquired)", "Failure (Closed)"],
        default=["Success (Acquired)", "Failure (Closed)"]
    )
    
    # Year range filter - fixed to 2000-2013
    min_year = 2000
    max_year = 2013
    year_range = st.sidebar.slider(
        "Founded Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # State filter
    state_options = sorted(df_raw['state_code'].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect(
        "States",
        options=state_options,
        default=state_options
    )
    
    # Industry category filter
    category_cols = ['is_software', 'is_web', 'is_mobile', 'is_enterprise', 
                    'is_advertising', 'is_gamesvideo', 'is_ecommerce', 
                    'is_biotech', 'is_consulting']
    category_cols = [col for col in category_cols if col in df_processed.columns]
    
    category_labels = {
        'is_software': 'Software',
        'is_web': 'Web',
        'is_mobile': 'Mobile',
        'is_enterprise': 'Enterprise',
        'is_advertising': 'Advertising',
        'is_gamesvideo': 'Games/Video',
        'is_ecommerce': 'E-commerce',
        'is_biotech': 'Biotech',
        'is_consulting': 'Consulting'
    }
    
    selected_categories = st.sidebar.multiselect(
        "Industry Categories",
        options=[category_labels.get(col, col) for col in category_cols],
        default=[category_labels.get(col, col) for col in category_cols]
    )
    
    # Funding range filter
    min_funding = int(df_processed['funding_total_usd'].min())
    max_funding = int(df_processed['funding_total_usd'].max())
    funding_range = st.sidebar.slider(
        "Total Funding (USD)",
        min_value=min_funding,
        max_value=max_funding,
        value=(min_funding, max_funding),
        step=100000
    )
    
    # Company age filter
    min_age = float(df_processed['company_age'].min())
    max_age = float(df_processed['company_age'].max())
    age_range = st.sidebar.slider(
        "Company Age (Years)",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=0.5
    )
    
    # Relationships filter
    min_relationships = int(df_processed['relationships'].min())
    max_relationships = int(df_processed['relationships'].max())
    relationships_range = st.sidebar.slider(
        "Number of Relationships",
        min_value=min_relationships,
        max_value=max_relationships,
        value=(min_relationships, max_relationships)
    )
    
    # Funding rounds filter
    min_rounds = int(df_processed['funding_rounds'].min())
    max_rounds = int(df_processed['funding_rounds'].max())
    rounds_range = st.sidebar.slider(
        "Funding Rounds",
        min_value=min_rounds,
        max_value=max_rounds,
        value=(min_rounds, max_rounds)
    )
    
    # Milestones filter
    has_milestones_filter = st.sidebar.selectbox(
        "Has Milestones",
        options=["All", "Yes", "No"],
        index=0
    )
    
    # VC/Investment type filters
    has_vc = st.sidebar.checkbox("Has VC Funding", value=False)
    filter_vc = st.sidebar.checkbox("Filter by VC Funding", value=False)
    
    has_angel = st.sidebar.checkbox("Has Angel Funding", value=False)
    filter_angel = st.sidebar.checkbox("Filter by Angel Funding", value=False)
    
    # ==================== APPLY FILTERS ====================
    filtered_raw = df_raw.copy()
    filtered_processed = df_processed_with_year.copy()
    
    # Success status filter
    if "Success (Acquired)" not in success_filter:
        filtered_raw = filtered_raw[filtered_raw['status'] != 'acquired']
        filtered_processed = filtered_processed[filtered_processed['success'] != 1]
    if "Failure (Closed)" not in success_filter:
        filtered_raw = filtered_raw[filtered_raw['status'] != 'closed']
        filtered_processed = filtered_processed[filtered_processed['success'] != 0]
    
    # Year filter
    filtered_raw = filtered_raw[
        (filtered_raw['founded_year'] >= year_range[0]) & 
        (filtered_raw['founded_year'] <= year_range[1])
    ]
    filtered_processed = filtered_processed[
        (filtered_processed['founded_year'] >= year_range[0]) & 
        (filtered_processed['founded_year'] <= year_range[1])
    ]
    
    # State filter
    if selected_states:
        filtered_raw = filtered_raw[filtered_raw['state_code'].isin(selected_states)]
        # Match indices for processed data
        matching_indices = filtered_raw.index.intersection(filtered_processed.index)
        filtered_processed = filtered_processed.loc[matching_indices]
    
    # Category filter
    if selected_categories:
        selected_category_cols = [col for col, label in category_labels.items() 
                                 if label in selected_categories and col in filtered_processed.columns]
        if selected_category_cols:
            # Filter where at least one selected category is True
            category_mask = filtered_processed[selected_category_cols].any(axis=1)
            filtered_processed = filtered_processed[category_mask]
            filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_raw.index)]
    
    # Funding range filter
    filtered_processed = filtered_processed[
        (filtered_processed['funding_total_usd'] >= funding_range[0]) &
        (filtered_processed['funding_total_usd'] <= funding_range[1])
    ]
    filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_raw.index)]
    
    # Age filter
    filtered_processed = filtered_processed[
        (filtered_processed['company_age'] >= age_range[0]) &
        (filtered_processed['company_age'] <= age_range[1])
    ]
    filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_raw.index)]
    
    # Relationships filter
    filtered_processed = filtered_processed[
        (filtered_processed['relationships'] >= relationships_range[0]) &
        (filtered_processed['relationships'] <= relationships_range[1])
    ]
    filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_raw.index)]
    
    # Rounds filter
    filtered_processed = filtered_processed[
        (filtered_processed['funding_rounds'] >= rounds_range[0]) &
        (filtered_processed['funding_rounds'] <= rounds_range[1])
    ]
    filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_processed.index)]
    
    # Milestones filter
    if has_milestones_filter == "Yes":
        filtered_processed = filtered_processed[filtered_processed['has_milestones'] == 1]
        filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_raw.index)]
    elif has_milestones_filter == "No":
        filtered_processed = filtered_processed[filtered_processed['has_milestones'] == 0]
        filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_raw.index)]
    
    # VC/Angel filters
    if filter_vc and 'has_VC' in filtered_processed.columns:
        filtered_processed = filtered_processed[filtered_processed['has_VC'] == (1 if has_vc else 0)]
        filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_raw.index)]
    
    if filter_angel and 'has_angel' in filtered_processed.columns:
        filtered_processed = filtered_processed[filtered_processed['has_angel'] == (1 if has_angel else 0)]
        filtered_raw = filtered_raw.loc[filtered_processed.index.intersection(filtered_raw.index)]
    
    # ==================== OVERVIEW SECTION ====================
    st.markdown('<h2 class="section-title">Dataset Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Startups", 923)
    with col2:
        success_rate = filtered_processed['success'].mean() * 100 if len(filtered_processed) > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col3:
        median_funding = filtered_processed['funding_total_usd'].median() / 1e6 if len(filtered_processed) > 0 else 0
        st.metric("Median Funding (M)", f"${median_funding:.2f}M")
    with col4:
        avg_relationships = filtered_processed['relationships'].mean() if len(filtered_processed) > 0 else 0
        st.metric("Avg Relationships", f"{avg_relationships:.1f}")
    
    st.markdown("---")
    
    # ==================== GEOGRAPHIC MAP ====================
    st.markdown('<h2 class="section-title">📍 Startup Geographic Distribution</h2>', unsafe_allow_html=True)
    
    try:
        map_df = filtered_raw[['latitude', 'longitude', 'status', 'name', 'city', 'state_code']].copy()
        map_df = map_df.dropna(subset=['latitude', 'longitude'])
        map_df = map_df[
            (map_df['latitude'] >= 24) & (map_df['latitude'] <= 50) &
            (map_df['longitude'] >= -130) & (map_df['longitude'] <= -65)
        ]
        
        if len(map_df) == 0:
            st.warning("No valid geographic data available for mapping.")
        else:
            map_df['success'] = map_df['status'].map({'acquired': 1, 'closed': 0})
            map_df['is_success'] = map_df['success'] == 1
            map_df['lat_rounded'] = map_df['latitude'].round(1)
            map_df['lon_rounded'] = map_df['longitude'].round(1)
            
            density_df = map_df.groupby(['lat_rounded', 'lon_rounded']).agg({
                'latitude': 'mean',
                'longitude': 'mean',
                'name': 'count',
                'success': 'mean'
            }).reset_index()
            density_df.columns = ['lat_rounded', 'lon_rounded', 'lat', 'lon', 'count', 'success_rate']
            
            map_type = st.radio(
                "Choose visualization type:",
                ["Individual Startups", "Success Rate Heatmap"],
                horizontal=True
            )
            
            if map_type == "Individual Startups":
                map_df['color'] = map_df['success'].apply(
                    lambda x: [0, 255, 0, 180] if x == 1 else [255, 0, 0, 180]
                )
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position=['longitude', 'latitude'],
                    get_color='color',
                    get_radius=2000,
                    radius_min_pixels=3,
                    radius_max_pixels=15,
                    pickable=True,
                    opacity=0.6,
                )
                
                view_state = pdk.ViewState(
                    latitude=map_df['latitude'].mean(),
                    longitude=map_df['longitude'].mean(),
                    zoom=3,
                    pitch=0,
                )
                
                st.pydeck_chart(pdk.Deck(
                    map_style='light',
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={
                        "html": "<b>Startup:</b> {name}<br/><b>City:</b> {city}<br/><b>State:</b> {state_code}<br/><b>Status:</b> {status}",
                        "style": {"color": "white", "backgroundColor": "rgba(0,0,0,0.7)", "padding": "5px"}
                    }
                ))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("🟢 **Green markers:** Acquired (Successful)")
                with col2:
                    st.caption("🔴 **Red markers:** Closed (Failed)")
            
            else:  # Success Rate Heatmap
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=density_df,
                    get_position=['lon', 'lat'],
                    get_weight='success_rate',
                    radius_pixels=60,
                    intensity=1,
                    threshold=0.05,
                    color_range=[
                        [255, 0, 0, 128],
                        [255, 128, 0, 192],
                        [255, 255, 0, 255],
                        [128, 255, 0, 192],
                        [0, 255, 0, 255]
                    ],
                )
                
                view_state = pdk.ViewState(
                    latitude=map_df['latitude'].mean(),
                    longitude=map_df['longitude'].mean(),
                    zoom=3,
                    pitch=0,
                )
                
                st.pydeck_chart(pdk.Deck(
                    map_style='light',
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={
                        "html": "<b>Success Rate:</b> {success_rate:.1%}<br/><b>Startups:</b> {count}<br/><b>Location:</b> {lat_rounded}°N, {lon_rounded}°W",
                        "style": {"color": "white", "backgroundColor": "rgba(0,0,0,0.7)", "padding": "5px"}
                    }
                ))
                
                st.caption("💡 **Heatmap showing success rate by location.** Greener areas indicate higher success rates.")
            
            st.markdown("#### 📊 Geographic Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Cities", map_df['city'].nunique())
            with col2:
                st.metric("Unique States", map_df['state_code'].nunique())
            with col3:
                top_state = map_df['state_code'].value_counts().index[0] if len(map_df) > 0 else "N/A"
                st.metric("Top State", top_state)
    
    except Exception as e:
        st.error(f"Error creating map visualization: {str(e)}")
        st.info("💡 **Tip:** Make sure pydeck is installed: `pip install pydeck`")
    
    st.markdown("---")
    
    # ==================== SUCCESS TRENDS ====================
    st.markdown('<h2 class="section-title">📈 Success Trends Over Time (2000-2013)</h2>', unsafe_allow_html=True)
    
    yearly_stats = filtered_raw.groupby('founded_year').agg({
        'status': lambda x: (x == 'acquired').sum() / len(x) * 100
    }).reset_index()
    yearly_stats.columns = ['Year', 'Success Rate (%)']
    yearly_stats = yearly_stats.dropna()
    yearly_stats = yearly_stats[yearly_stats['Year'] >= 2000]
    
    if len(yearly_stats) > 0:
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
    else:
        st.info("No data available for the selected filters in this time period.")
    
    st.markdown("---")
    
    # ==================== FEATURE DISTRIBUTIONS ====================
    st.markdown('<h2 class="section-title">📊 Key Feature Distributions</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Funding Total (Log Scale)")
        if len(filtered_processed) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            log_funding = np.log1p(filtered_processed['funding_total_usd'])
            ax.hist(log_funding, bins=40, color='#2ca02c', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Log(Funding Total USD)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Total Funding (Log Scale)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.markdown("#### Company Age Distribution")
        if len(filtered_processed) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(filtered_processed['company_age'].dropna(), bins=30, color='#ff7f0e', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Company Age (Years)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Company Age', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Funding Rounds")
        if len(filtered_processed) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(filtered_processed['funding_rounds'], bins=20, color='#d62728', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Funding Rounds', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Funding Rounds', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.markdown("#### Relationships")
        if len(filtered_processed) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(filtered_processed['relationships'], bins=30, color='#9467bd', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Relationships', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Distribution of Relationships', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # ==================== CORRELATION HEATMAP ====================
    st.markdown('<h2 class="section-title">🔥 Correlation Heatmap - Key Numeric Features</h2>', unsafe_allow_html=True)
    
    if len(filtered_processed) > 0:
        numeric_cols = [
            'company_age', 'time_to_first_funding', 'funding_duration',
            'funding_rounds', 'funding_total_usd', 'avg_participants',
            'relationships', 'milestones', 'age_first_funding_year', 'age_last_funding_year'
        ]
        numeric_cols = [col for col in numeric_cols if col in filtered_processed.columns]
        
        corr_matrix = filtered_processed[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Key Numeric Features', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # ==================== SUCCESS BY CATEGORY ====================
    st.markdown('<h2 class="section-title">🏢 Success Rate by Category</h2>', unsafe_allow_html=True)
    
    if len(filtered_processed) > 0 and len(category_cols) > 0:
        category_success = []
        for col in category_cols:
            category_name = category_labels.get(col, col.replace('is_', '').title())
            category_data = filtered_processed[filtered_processed[col] == 1]
            if len(category_data) > 0:
                success_rate = category_data['success'].mean() * 100
                category_success.append({'Category': category_name, 'Success Rate (%)': success_rate, 'Count': len(category_data)})
        
        if category_success:
            category_df = pd.DataFrame(category_success).sort_values('Success Rate (%)', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(category_df['Category'], category_df['Success Rate (%)'], color='#1f77b4')
            ax.set_xlabel('Success Rate (%)', fontsize=12)
            ax.set_title('Success Rate by Industry Category', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.dataframe(category_df, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== GEOGRAPHIC DISTRIBUTION ====================
    st.markdown('<h2 class="section-title">🗺️ Geographic Distribution</h2>', unsafe_allow_html=True)
    
    if len(filtered_processed) > 0:
        state_cols = ['is_CA', 'is_NY', 'is_MA', 'is_TX']
        state_cols = [col for col in state_cols if col in filtered_processed.columns]
        
        state_success = []
        state_names = {'is_CA': 'California', 'is_NY': 'New York', 
                      'is_MA': 'Massachusetts', 'is_TX': 'Texas'}
        
        for col in state_cols:
            state_name = state_names.get(col, col.replace('is_', ''))
            state_data = filtered_processed[filtered_processed[col] == 1]
            if len(state_data) > 0:
                success_rate = state_data['success'].mean() * 100
                state_success.append({'State': state_name, 'Success Rate (%)': success_rate, 'Count': len(state_data)})
        
        if state_success:
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

