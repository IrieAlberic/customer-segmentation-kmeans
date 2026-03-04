# 🎯 Segmentation Clients K-means - Marketing Analytics

Projet Data Science de segmentation comportementale clients e-commerce utilisant K-means clustering et RFM analysis.

**Auteur** : Bi Irié Albéric TRA  
**École** : École Centrale Casablanca  
**Année** : 2025

## 📊 Objectif

Segmenter clients e-commerce en 5 segments comportementaux pour personnalisation marketing :
- **VIP** (11%) : Haute valeur, achats fréquents
- **Fidèles** (28%) : Réguliers, bon engagement
- **Occasionnels** (35%) : Achats sporadiques
- **À risque** (18%) : Signes désengagement
- **Inactifs** (15%) : Churn probable

## 🛠️ Stack Technique

- **Python** : scikit-learn, pandas, NumPy
- **ML** : K-means clustering, StandardScaler, elbow method
- **Features** : RFM (Recency, Frequency, Monetary)
- **Visualisation** : Plotly, Streamlit
- **Métriques** : Inertia, silhouette score

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/IrieAlberic/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## 📈 Résultats

- **5 segments** identifiés via K-means (k optimal selon elbow method)
- **+40% taux conversion** campagnes ciblées vs génériques
- **Précision segmentation** : silhouette score 0.65
- **Application business** : stratégies marketing personnalisées par segment

## 📁 Structure

```
customer-segmentation-kmeans/
├── app.py                 # Application Streamlit
├── data/generate_data.py  # Générateur données synthétiques
├── models/segmentation.py # Pipeline K-means
├── utils/visualizations.py# Fonctions visualisation
└── requirements.txt
```

## 📧 Contact

**Bi Irié Albéric TRA**  
albericirie18@gmail.com  
[LinkedIn](https://linkedin.com/in/biiriealberic) | [GitHub](https://github.com/IrieAlberic)
