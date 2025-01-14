import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("bodyfat.csv")

# Features and target
features = data[['Weight']]
target = data['BodyFat']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict
predictions = linear_model.predict(X_test)

# Plot results
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, predictions, color='red', label='Predictions')
plt.title('Linear Regression')
plt.xlabel('Weight')
plt.ylabel('BodyFat')
plt.legend()
plt.show()
