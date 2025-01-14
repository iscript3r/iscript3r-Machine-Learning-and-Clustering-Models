#3 DBSCAN Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#We load our medical data set 
df = pd.read_csv("patient_priority.csv")

#We selected the values that we need for clusstering 
selected_columns = ['age', 'gender', 'chest pain type', 'blood pressure', 'cholesterol', 'max heart rate', 'hypertension', 'heart_disease']
X = df[selected_columns]

#Handling missing values
X.dropna(inplace=True)

#Standardizing the features using StandardScaler() that is provided by sklearn 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Clustering with DBSCAN 
dbscan = DBSCAN(eps=5, min_samples=5)
cluster_labels = dbscan.fit_predict(X_scaled)

#Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('DBSCAN Clustering ')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()
