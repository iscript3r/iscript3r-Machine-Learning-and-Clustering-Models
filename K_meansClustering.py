import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv("patient_dataset.csv")

# Select relevant columns
features = ['age', 'gender', 'chest_pain_type', 'blood_pressure', 'cholesterol', 
            'max_heart_rate', 'exercise_angina', 'plasma_glucose', 'skin_thickness', 
            'insulin', 'bmi', 'diabetes_pedigree', 'hypertension', 'heart_disease', 
            'residence_type', 'smoking_status']
X = data[features]

# Handle missing values
X.dropna(inplace=True)

# Encode categorical features
categorical_cols = ['gender', 'chest_pain_type', 'residence_type', 'smoking_status']
transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_cols)],
                                 remainder='passthrough')
X_encoded = transformer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Perform KMeans clustering
model = KMeans(n_clusters=3, random_state=42)
cluster_labels = model.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
