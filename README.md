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

## 📈 Structure des Données

### Génération des Données

Les données sont **synthétiques** et générées automatiquement via `data/generate_data.py`. Chaque run crée un dataset frais avec corrélations réalistes.

### Colonnes (Features RFM)

| Colonne | Type | Description | Plage | Exemple |
|---------|------|-------------|-------|---------|
| `customer_id` | String | Identifiant unique client | CUST_00001 à CUST_N | CUST_05234 |
| `recency` | Integer | Jours depuis dernier achat | 1 à 365 | 23 |
| `frequency` | Integer | Nombre total d'achats | 1 à 50 | 12 |
| `monetary` | Float | Montant total dépensé (MAD) | 50 à 20,000 | 4,532.50 |
| `avg_order_value` | Float | Panier moyen (MAD) | 50 à 20,000 | 377.71 |
| `days_since_registration` | Integer | Jours depuis inscription | 30 à 730 | 245 |

### Exemple de DataFrame

```
     customer_id  recency  frequency  monetary  avg_order_value  days_since_registration
0    CUST_00001       45           8   3,200.50          400.06                       180
1    CUST_00002       12          25  12,450.75          498.03                       650
2    CUST_00003      120           2     450.00          225.00                        45
3    CUST_00004       78          15   8,900.25          593.35                       520
```

### Distribution des Données

**Recency (Jours)** :
- Moyenne : ~60 jours
- Distribution : Gamma (shape=2, scale=30)
- Signification : Clients actifs = Recency faible

**Frequency (Achats)** :
- Moyenne : ~5 achats
- Distribution : Poisson (λ=5)
- Signification : Plus d'achats = Client plus engagé

**Monetary (Dépenses)** :
- Moyenne : ~3,000 MAD
- Distribution : Corrélée à `frequency`
- Signification : Plus de dépenses = CLV élevé

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

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

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
