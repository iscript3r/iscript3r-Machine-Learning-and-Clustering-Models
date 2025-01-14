import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv('diabetes.csv')
cols_with_zeros = ['Glucose', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# Define features and labels
X = df[['Glucose', 'Insulin', 'BMI']]
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluate RandomForest with varying estimators
accuracy_results = []
for estimators in range(5, 16):
    model = RandomForestClassifier(n_estimators=estimators)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, predictions)
    accuracy_results.append((estimators, acc))
    
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(f"Confusion Matrix (n_estimators={estimators})")
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

best_n, best_acc = max(accuracy_results, key=lambda x: x[1])
print(f"Best Estimator: {best_n}, Accuracy: {best_acc:.2f}")
