"""
Générateur de données synthétiques clients e-commerce
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(n_customers=5000, seed=42):
    """
    Génère données synthétiques clients avec métriques RFM
    
    Returns:
        DataFrame avec colonnes : customer_id, recency, frequency, monetary
    """
    np.random.seed(seed)
    
    # Génération IDs
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    
    # Recency (jours depuis dernier achat) - distribution gamma
    recency = np.random.gamma(shape=2, scale=30, size=n_customers).astype(int)
    recency = np.clip(recency, 1, 365)
    
    # Frequency (nombre achats) - distribution poisson
    frequency = np.random.poisson(lam=5, size=n_customers)
    frequency = np.clip(frequency, 1, 50)
    
    # Monetary (montant total dépensé) - corrélé avec frequency
    base_monetary = frequency * np.random.gamma(shape=3, scale=100, size=n_customers)
    noise = np.random.normal(0, 50, size=n_customers)
    monetary = base_monetary + noise
    monetary = np.clip(monetary, 50, 20000)
    
    # Features additionnelles
    avg_order_value = monetary / frequency
    days_since_registration = np.random.randint(30, 730, size=n_customers)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary.round(2),
        'avg_order_value': avg_order_value.round(2),
        'days_since_registration': days_since_registration
    })
    
    return df

if __name__ == "__main__":
    df = generate_customer_data()
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nRFM Summary:\n{df[['recency', 'frequency', 'monetary']].describe()}")
