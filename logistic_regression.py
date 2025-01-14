import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('diabetes.csv')
invalid_cols = ['Glucose', 'Insulin', 'BMI']
data[invalid_cols] = data[invalid_cols].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

X = data[['Glucose', 'Insulin', 'BMI']]
y = data['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluate Logistic Regression with varying hyperparameters
accuracy_results = []
c_values = [0.01, 0.1, 1, 10, 100]
penalties = ['l1', 'l2', 'none']
solvers = ['liblinear', 'saga']

for c in c_values:
    for penalty in penalties:
        for solver in solvers:
            if penalty == 'l1' and solver != 'liblinear':
                continue
            if penalty == 'none' and solver != 'lbfgs':
                continue
            try:
                log_model = LogisticRegression(C=c, penalty=penalty, solver=solver, max_iter=1000)
                log_model.fit(X_train_scaled, y_train)
                predictions = log_model.predict(X_test_scaled)

                acc = accuracy_score(y_test, predictions)
                accuracy_results.append((c, penalty, solver, acc))

                cm = confusion_matrix(y_test, predictions)
                sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
                plt.title(f"Logistic Confusion Matrix (C={c}, Penalty={penalty}, Solver={solver})")
                plt.show()
            except ValueError as e:
                print(f"Skipped combination C={c}, Penalty={penalty}, Solver={solver}: {e}")

best_config = max(accuracy_results, key=lambda x: x[3])
print(f"Best Config: C={best_config[0]}, Penalty={best_config[1]}, Solver={best_config[2]}, Accuracy={best_config[3]:.2f}")
