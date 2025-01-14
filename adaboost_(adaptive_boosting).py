import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess dataset
data = pd.read_csv('diabetes.csv')
zero_features = ['Glucose', 'Insulin', 'BMI']
data[zero_features] = data[zero_features].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

X = data[['Glucose', 'Insulin', 'BMI']]
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluate AdaBoost with varying estimators
results = []
n_estimators_range = range(50, 150, 10)
for n_estimators in n_estimators_range:
    ada = AdaBoostClassifier(n_estimators=n_estimators)
    ada.fit(X_train_scaled, y_train)
    predictions = ada.predict(X_test_scaled)

    acc = accuracy_score(y_test, predictions)
    results.append((n_estimators, acc))

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(f"AdaBoost Confusion Matrix (n_estimators={n_estimators})")
    plt.show()

best_n_estimators, best_acc = max(results, key=lambda x: x[1])
print(f"Best n_estimators: {best_n_estimators}, Accuracy: {best_acc:.2f}")
