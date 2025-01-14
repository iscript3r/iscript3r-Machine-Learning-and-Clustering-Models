# -*- coding: utf-8 -*-
"""K-Nearest Neighbors (KNN) classifier.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MBX4CR4-VHWm85z-fCVsn1R0LNuCsgFo
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('diabetes.csv')

# clean data by replacing invalid data with the median
columns_with_invalid_zeros = ['Glucose','Insulin', 'BMI']
data[columns_with_invalid_zeros] = data[columns_with_invalid_zeros].replace(0, np.nan)

data.fillna(data.median(), inplace=True)

X = data[['Glucose', 'Insulin', 'BMI']]
y = data['Outcome']



# Splitting the dataset into the Training set and Test set using sklearn with a 20-80 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# List to store accuracies and configurations
knn_accuracies = []

# Parameter ranges
neighbors_range = range(1, 11)  # Trying 1 to 10 neighbors
weights_options = ['uniform', 'distance']
metrics_options = ['euclidean', 'manhattan']

# Nested loops to try different combinations of parameters
for n_neighbors in neighbors_range:
    for weights in weights_options:
        for metric in metrics_options:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Calculate accuracy and store results
            accuracy = accuracy_score(y_test, y_pred)
            knn_accuracies.append((n_neighbors, weights, metric, accuracy))
            title = f'KNN Neighbors={n_neighbors}, Weights={weights}, Metric={metric}, Acc={accuracy:.2f}'
            plot_confusion_matrix(confusion_matrix(y_test, y_pred), title)

# Determine the configuration with the highest accuracy
optimal_settings = max(knn_accuracies, key=lambda x: x[3])
print(f"Optimal Configuration: {optimal_settings[0]} Neighbors, Weights: {optimal_settings[1]}, Metric: {optimal_settings[2]} with Accuracy: {optimal_settings[3]:.2f}")