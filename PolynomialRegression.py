import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Load the dataset
dataset = pd.read_csv("bodyfat.csv")

# Select features and target
features = dataset[['Weight']]
target = dataset['BodyFat']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a pipeline for polynomial regression
degree = 2
regression_model = Pipeline([('poly_features', PolynomialFeatures(degree=degree)),
                              ('linear_model', LinearRegression())])

# Train the model
regression_model.fit(X_train, y_train)

# Generate predictions
predictions = regression_model.predict(X_test)

# Prepare data for plotting
sorted_data = sorted(zip(X_test.values, predictions))
sorted_X, sorted_predictions = zip(*sorted_data)

# Visualize results
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(sorted_X, sorted_predictions, color='red', label='Model Predictions')
plt.title(f'Polynomial Regression (Degree = {degree})')
plt.xlabel('Weight')
plt.ylabel('BodyFat')
plt.legend()
plt.show()
