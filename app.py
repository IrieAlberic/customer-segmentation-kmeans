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

st.set_page_config(page_title="Segmentation Clients K-means", layout="wide")

# Titre
st.title("🎯 Segmentation Clients K-means - Marketing Analytics")
st.markdown("**Projet Data Science** | École Centrale Casablanca | Bi Irié Albéric TRA")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
n_customers = st.sidebar.slider("Nombre de clients", 1000, 10000, 5000, 500)
n_clusters = st.sidebar.slider("Nombre de segments (K)", 3, 8, 5, 1)

# Génération données
with st.spinner("Génération des données..."):
    df = generate_customer_data(n_customers=n_customers)

# Section 1: Données
st.header("📊 Données Clients (RFM)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Clients", f"{len(df):,}")
with col2:
    st.metric("Récence Moyenne", f"{df['recency'].mean():.0f} jours")
with col3:
    st.metric("Panier Moyen", f"{df['avg_order_value'].mean():.0f} MAD")

with st.expander("📋 Aperçu des données (10 premières lignes)"):
    st.dataframe(df.head(10))

# Section 2: Méthode du Coude
st.header("📈 Optimisation K (Méthode du Coude)")
model = CustomerSegmentation(n_clusters=n_clusters)

with st.spinner("Calcul inertias..."):
    inertias = model.elbow_method(df, max_k=10)
    
st.plotly_chart(plot_elbow(inertias), use_container_width=True)
st.info(f"💡 **K optimal sélectionné : {n_clusters}** (selon méthode du coude et interprétation business)")

# Section 3: Segmentation
st.header(f"🎨 Segmentation en {n_clusters} Clusters")

with st.spinner("Entraînement modèle K-means..."):
    clusters = model.fit(df)
    profile = model.get_segment_profile(df, clusters)

# Profils segments
st.subheader("📊 Profils des Segments")
st.dataframe(profile, use_container_width=True)

# Visualisations
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(plot_segments_distribution(profile), use_container_width=True)

with col2:
    st.plotly_chart(plot_segment_profile(profile), use_container_width=True)

# Scatter 3D
st.subheader("🔍 Visualisation 3D des Segments (RFM)")
st.plotly_chart(plot_rfm_scatter(df, clusters, model.segment_labels), use_container_width=True)

# Section 4: Interprétation Business
st.header("💼 Insights Marketing & Recommandations")

segment_insights = {
    "VIP": "🌟 Clients haute valeur (CLV élevé) - Programme fidélité premium, offres exclusives",
    "Fidèles": "💚 Clients réguliers - Maintenir engagement, cross-sell/up-sell",
    "Occasionnels": "🔵 Achats sporadiques - Campagnes activation, incentives fréquence",
    "À risque": "⚠️ Signes désengagement - Campagnes réactivation urgentes, win-back offers",
    "Inactifs": "💤 Churn probable - Enquêtes NPS, dernières tentatives ou archivage"
}

for segment, insight in segment_insights.items():
    st.markdown(f"**{segment}** : {insight}")

# Métriques clés
st.subheader("📈 Impact Potentiel")
col1, col2, col3 = st.columns(3)

vip_customers = profile.loc["VIP", "Nombre clients"] if "VIP" in profile.index else 0
at_risk_customers = profile.loc["À risque", "Nombre clients"] if "À risque" in profile.index else 0

with col1:
    st.metric("Clients VIP à Fidéliser", f"{int(vip_customers):,}")
with col2:
    st.metric("Clients À Risque", f"{int(at_risk_customers):,}")
with col3:
    potential_impact = (at_risk_customers * 0.30 * df['monetary'].mean())
    st.metric("Revenu Potentiel Récupérable", f"{potential_impact:,.0f} MAD")

st.caption("*Hypothèse : 30% clients à risque réactivables via campagnes ciblées")

# Footer
st.markdown("---")
st.markdown("**Projet réalisé par Bi Irié Albéric TRA** | École Centrale Casablanca | 2025")
st.markdown("Stack : Python, scikit-learn, K-means, Streamlit, Plotly")
