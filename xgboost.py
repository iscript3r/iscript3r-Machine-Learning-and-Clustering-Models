import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# Load and preprocess data
data = pd.read_csv('diabetes.csv')
zero_cols = ['Glucose', 'Insulin', 'BMI']
data[zero_cols] = data[zero_cols].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

X = data[['Glucose', 'Insulin', 'BMI']]
y = data['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluate XGBoost with varying hyperparameters
accuracy_results = []

for lr in np.arange(0.01, 0.2, 0.05):
    for estimators in range(50, 150, 10):
        for depth in range(2, 7):
            xgb_model = xgb.XGBClassifier(learning_rate=lr, n_estimators=estimators, max_depth=depth, use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            predictions = xgb_model.predict(X_test)
            
            acc = accuracy_score(y_test, predictions)
            accuracy_results.append((lr, estimators, depth, acc))
            
            cm = confusion_matrix(y_test, predictions)
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
            plt.title(f"XGBoost Confusion Matrix (LR={lr}, Est={estimators}, Depth={depth})")
            plt.show()

# Find best configuration
best_config = max(accuracy_results, key=lambda x: x[3])
print(f"Best Config: LR={best_config[0]}, Estimators={best_config[1]}, Depth={best_config[2]}, Accuracy={best_config[3]:.2f}")
