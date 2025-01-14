import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv("patient_priority.csv")

# Select features
features = ['age', 'gender', 'chest pain type', 'blood pressure', 'cholesterol', 
            'max heart rate', 'exercise angina', 'plasma glucose', 'skin_thickness', 
            'insulin', 'bmi', 'diabetes_pedigree', 'hypertension', 'heart_disease']
X = data[features]

# Handle missing values
X.dropna(inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Agglomerative Clustering
agg_model = AgglomerativeClustering(n_clusters=3)
cluster_labels = agg_model.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
