import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
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

# Evaluate MLPClassifier with varying configurations
accuracy_results = []
layer_sizes = [(50,), (100,), (50, 50)]
activations = ['relu', 'tanh']
solvers = ['adam', 'sgd']

for layers in layer_sizes:
    for activation in activations:
        for solver in solvers:
            nn_model = MLPClassifier(hidden_layer_sizes=layers, activation=activation, solver=solver, max_iter=1000)
            nn_model.fit(X_train_scaled, y_train)
            predictions = nn_model.predict(X_test_scaled)

            acc = accuracy_score(y_test, predictions)
            accuracy_results.append((layers, activation, solver, acc))

            cm = confusion_matrix(y_test, predictions)
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
            plt.title(f"Neural Network (Layers={layers}, Activation={activation}, Solver={solver})")
            plt.show()

best_config = max(accuracy_results, key=lambda x: x[3])
print(f"Best Config: Layers={best_config[0]}, Activation={best_config[1]}, Solver={best_config[2]}, Accuracy={best_config[3]:.2f}")
