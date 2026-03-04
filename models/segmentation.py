"""
Pipeline segmentation K-means avec RFM analysis
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class CustomerSegmentation:
    """Classe pour segmentation clients K-means"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.segment_labels = {
            0: "VIP",
            1: "Fidèles", 
            2: "Occasionnels",
            3: "À risque",
            4: "Inactifs"
        }
        
    def fit(self, df):
        """
        Entraîne le modèle K-means sur données RFM
        
        Args:
            df: DataFrame avec colonnes recency, frequency, monetary
        """
        # Sélection features RFM
        X = df[['recency', 'frequency', 'monetary']].values
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        # K-means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        return clusters
    
    def predict(self, df):
        """Prédiction segments pour nouveaux clients"""
        X = df[['recency', 'frequency', 'monetary']].values
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def get_segment_profile(self, df, clusters):
        """
        Calcule profil moyen de chaque segment
        
        Returns:
            DataFrame avec statistiques par segment
        """
        df_copy = df.copy()
        df_copy['segment'] = clusters
        
        # Mapping vers labels texte
        df_copy['segment_label'] = df_copy['segment'].map(self.segment_labels)
        
        # Calcul profils
        profile = df_copy.groupby('segment_label').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'avg_order_value': 'mean'
        }).round(2)
        
        profile.columns = ['Nombre clients', 'Récence moy', 'Fréquence moy', 
                          'Montant total moy', 'Panier moyen']
        
        # Pourcentage
        profile['Pourcentage'] = (profile['Nombre clients'] / len(df) * 100).round(1)
        
        return profile.sort_values('Montant total moy', ascending=False)
    
    def elbow_method(self, df, max_k=10):
        """
        Calcule inertia pour méthode du coude
        
        Returns:
            Liste des inertias pour k=1 à max_k
        """
        X = df[['recency', 'frequency', 'monetary']].values
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        return inertias

if __name__ == "__main__":
    from data.generate_data import generate_customer_data
    
    # Test
    df = generate_customer_data(n_customers=1000)
    
    model = CustomerSegmentation(n_clusters=5)
    clusters = model.fit(df)
    
    profile = model.get_segment_profile(df, clusters)
    print("Profils segments:")
    print(profile)
