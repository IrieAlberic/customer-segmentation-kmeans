"""
Fonctions visualisation pour segmentation clients
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_elbow(inertias):
    """Graphique méthode du coude"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(inertias) + 1)),
        y=inertias,
        mode='lines+markers',
        marker=dict(size=10, color='#004F90'),
        line=dict(width=3, color='#004F90')
    ))
    
    fig.update_layout(
        title="Méthode du Coude - Choix optimal de K",
        xaxis_title="Nombre de clusters (K)",
        yaxis_title="Inertia",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_segments_distribution(profile_df):
    """Graphique distribution segments"""
    fig = px.pie(
        values=profile_df['Nombre clients'],
        names=profile_df.index,
        title="Distribution des Segments Clients",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def plot_rfm_scatter(df, clusters, segment_labels):
    """Scatter 3D RFM par segment"""
    df_plot = df.copy()
    df_plot['segment'] = clusters
    df_plot['segment_label'] = df_plot['segment'].map(segment_labels)
    
    fig = px.scatter_3d(
        df_plot,
        x='recency',
        y='frequency',
        z='monetary',
        color='segment_label',
        title="Visualisation 3D des Segments (RFM)",
        labels={'recency': 'Récence (jours)', 
                'frequency': 'Fréquence', 
                'monetary': 'Montant (MAD)'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(height=600)
    
    return fig

def plot_segment_profile(profile_df):
    """Graphique barres profils segments"""
    fig = go.Figure()
    
    metrics = ['Récence moy', 'Fréquence moy', 'Montant total moy']
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=profile_df.index,
            y=profile_df[metric],
        ))
    
    fig.update_layout(
        title="Profils Moyens par Segment",
        xaxis_title="Segment",
        yaxis_title="Valeur",
        barmode='group',
        template="plotly_white",
        height=400
    )
    
    return fig
