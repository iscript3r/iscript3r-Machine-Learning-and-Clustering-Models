#3 DecisionTreeRegressor

#Importing our libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#We load our medical data set 
df = pd.read_csv("bodyfat.csv")

#we extract both columns of the chosen feature (Weight) , and our target that we want to predict (Body fat)
X = df[['Weight']]  # Our feature
y = df['BodyFat']  # Target

#we use a utility in scikit-learn that is usually used to split the dataset into two one for training the machine and another for testing it .
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Creating the Decision Tree Regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#Predictions with the model
y_pred = model.predict(X_test)

#Sorting test data for plotting
sorted_indices = np.argsort(X_test.values.flatten())
X_test_sorted = X_test.values[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

#Plotting our results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test_sorted, y_pred_sorted, color='green', label='Predicted', linewidth=2)
plt.title('Decision Tree Regression')
plt.xlabel('Weight')
plt.ylabel('BodyFat')
plt.legend()
plt.show()
