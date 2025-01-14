# Machine Learning and Clustering Models

This repository contains various machine learning models and clustering algorithms implemented in Python. The project explores different techniques for regression, classification, and clustering using real-world datasets like body fat percentage, diabetes diagnostics, and patient priority analysis.

### **Machine Learning Models**
- `LinearRegression.py`: Implements linear regression to predict body fat percentage based on weight.
- `PolynomialRegression.py`: Polynomial regression for predicting body fat with higher-order relationships.
- `logistic_regression.py`: Logistic regression for classifying diabetes outcomes.
- `naive_bayes.py`: Naive Bayes classifier for diabetes diagnostics.
- `support_vector_machines.py`: SVM model with hyperparameter tuning for binary classification.
- `randomforest.py`: Random Forest classifier for diabetes outcomes.
- `gradient_boosting_machines_(gbm).py`: Gradient boosting model with learning rate optimization.
- `adaboost_(adaptive_boosting).py`: AdaBoost classifier for improving classification accuracy.
- `neural_networks.py`: Multi-layer perceptron (MLP) classifier for diabetes predictions.

### **Clustering Algorithms**
- `K_meansClustering.py`: K-means clustering to analyze patient data.
- `DBSCANClustering.py`: Density-based clustering of patient priority datasets.
- `AgglomerativeClustering.py`: Hierarchical clustering with PCA for visualization.

### **Other Scripts**
- `decision_trees.py`: Decision tree classifier with depth tuning.
- `DecisionTreeRegressor.py`: Decision tree regression for body fat predictions.

### **Datasets**
- `bodyfat.csv`: Contains data for body fat predictions.
- `diabetes.csv`: Dataset for diabetes classification.
- `patient_dataset.csv`: Contains patient data for clustering analysis.
- `patient_priority.csv`: Dataset for clustering and priority assessment.

## Requirements

The project uses Python 3.x and requires the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
