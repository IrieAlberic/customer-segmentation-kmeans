"""
Application Streamlit - Segmentation Clients K-means
Auteur: Bi Irié Albéric TRA
"""
import streamlit as st
import pandas as pd
from data.generate_data import generate_customer_data
from models.segmentation import CustomerSegmentation
from utils.visualizations import (
    plot_elbow, plot_segments_distribution, 
    plot_rfm_scatter, plot_segment_profile
)

st.set_page_config(
    page_title="Customer Segmentation K-means",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé - Design Premium
st.markdown("""
<style>
    /* Main content styling */
    .main {
        padding: 2rem 3rem;
    }
    
    /* Header styling */
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e8eaed;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #e8eaed;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #30363d;
        padding-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.3rem;
        font-weight: 500;
        color: #c9d1d9;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1f26;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar {
        background-color: #0f1419;
        padding: 2rem 1.5rem;
    }
    
    .sidebar-content {
        color: #e8eaed;
    }
    
    /* Text styling */
    p, span {
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1f77b4;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2a8bd9;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    
    /* Info boxes */
    [data-testid="stInfo"] {
        background-color: #1a1f26;
        border-left: 4px solid #1f77b4;
        border-radius: 6px;
        color: #c9d1d9;
    }
    
    /* Error boxes */
    [data-testid="stError"] {
        background-color: #1a1f26;
        border-left: 4px solid #f85149;
        border-radius: 6px;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# Titre
st.markdown("<h1>Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.95rem; color: #8b949e; margin-top: -1rem;'>RFM-Based K-means Clustering & Marketing Analytics | École Centrale Casablanca</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.markdown("<h3 style='margin-top: 0;'>Configuration</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    n_customers = st.slider("Number of Customers", 1000, 10000, 5000, 500)
    n_clusters = st.slider("Number of Segments (K)", 3, 8, 5, 1)
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.85rem; color: #8b949e;'>Premium Analytics Engine | v1.0</p>", unsafe_allow_html=True)

# Génération données
with st.spinner("Génération des données..."):
    df = generate_customer_data(n_customers=n_customers)

# Section 1: Données
st.markdown("<h2>Dataset Overview - RFM Analysis</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", f"{len(df):,}")
with col2:
    st.metric("Average Recency", f"{df['recency'].mean():.0f} days")
with col3:
    st.metric("Average Order Value", f"{df['avg_order_value'].mean():.0f} MAD")

with st.expander("Data Preview"):
    st.dataframe(df.head(10), use_container_width=True)

# Section 2: Méthode du Coude
st.markdown("<h2>K Optimization - Elbow Method</h2>", unsafe_allow_html=True)

model = CustomerSegmentation(n_clusters=n_clusters)

with st.spinner("Computing inertia values..."):
    inertias = model.elbow_method(df, max_k=10)
    
st.plotly_chart(plot_elbow(inertias), use_container_width=True)
st.info(f"Optimal K selected: {n_clusters} (based on elbow method and business interpretation)")

# Section 3: Segmentation
st.markdown(f"<h2>Segmentation Results - {n_clusters} Clusters</h2>", unsafe_allow_html=True)

with st.spinner("Training K-means model..."):
    clusters = model.fit(df)
    profile = model.get_segment_profile(df, clusters)

# Profils segments
st.markdown("<h3>Segment Profiles</h3>", unsafe_allow_html=True)
st.dataframe(profile, use_container_width=True)

# Visualisations
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(plot_segments_distribution(profile), use_container_width=True)

with col2:
    st.plotly_chart(plot_segment_profile(profile), use_container_width=True)

# Scatter 3D
st.markdown("<h3>3D Visualization - RFM Space</h3>", unsafe_allow_html=True)
st.plotly_chart(plot_rfm_scatter(df, clusters, model.segment_labels), use_container_width=True)

# Section 4: Interprétation Business
st.markdown("<h2>Strategic Insights & Marketing Recommendations</h2>", unsafe_allow_html=True)

segment_insights = {
    "VIP": "High-value customers (premium CLV) - Loyalty programs, exclusive offers, dedicated support",
    "Fidèles": "Regular customers - Maintain engagement, cross-sell/up-sell opportunities",
    "Occasionnels": "Sporadic purchases - Activation campaigns, frequency incentives",
    "À risque": "Disengagement signals - Urgent reactivation campaigns, win-back offers",
    "Inactifs": "Probable churn - NPS surveys, last-ditch efforts or archival"
}

for segment, insight in segment_insights.items():
    st.markdown(f"**{segment}** — {insight}")

# Métriques clés
st.markdown("<h3>Impact Potential</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

vip_customers = profile.loc["VIP", "Nombre clients"] if "VIP" in profile.index else 0
at_risk_customers = profile.loc["À risque", "Nombre clients"] if "À risque" in profile.index else 0

with col1:
    st.metric("VIP Customers to Retain", f"{int(vip_customers):,}")
with col2:
    st.metric("At-Risk Customers", f"{int(at_risk_customers):,}")
with col3:
    potential_impact = (at_risk_customers * 0.30 * df['monetary'].mean())
    st.metric("Recoverable Revenue Potential", f"{potential_impact:,.0f} MAD")

st.caption("Assumption: 30% of at-risk customers recoverable through targeted campaigns")

# Footer
st.markdown("---")
st.markdown("<p style='font-size: 0.85rem; color: #8b949e; text-align: center;'>Developed by Bi Irié Albéric TRA | École Centrale Casablanca | 2025</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.85rem; color: #8b949e; text-align: center;'>Tech Stack: Python, scikit-learn, K-means, Streamlit, Plotly</p>", unsafe_allow_html=True)
