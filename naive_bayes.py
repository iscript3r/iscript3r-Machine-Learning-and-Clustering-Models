import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess dataset
data = pd.read_csv('diabetes.csv')
invalid_columns = ['Glucose', 'Insulin', 'BMI']
data[invalid_columns] = data[invalid_columns].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

X = data[['Glucose', 'Insulin', 'BMI']]
y = data['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluate Naive Bayes with different smoothing parameters
accuracy_results = []
smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

for smoothing in smoothing_values:
    nb_model = GaussianNB(var_smoothing=smoothing)
    nb_model.fit(X_train_scaled, y_train)
    predictions = nb_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, predictions)
    accuracy_results.append((smoothing, acc))

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(f"Naive Bayes Confusion Matrix (Smoothing={smoothing})")
    plt.show()

best_smoothing, best_acc = max(accuracy_results, key=lambda x: x[1])
print(f"Best Smoothing: {best_smoothing}, Accuracy: {best_acc:.2f}")
