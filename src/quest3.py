def simple_train_test_split(X, y, test_size=.3):
    n_training_samples = int((1.0 - test_size) * X.shape[0])

    X_train = X[:n_training_samples,:]
    y_train = y[:n_training_samples]

    X_test = X[n_training_samples:,:]
    y_test = y[n_training_samples:]

    return X_train, X_test, y_train, y_test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split

# Define simple_train_test_split function
def simple_train_test_split(X, y, test_size=.3):
    n_training_samples = int((1.0 - test_size) * X.shape[0])
    X_train = X[:n_training_samples,:]
    y_train = y[:n_training_samples]
    X_test = X[n_training_samples:,:]
    y_test = y[n_training_samples:]
    return X_train, X_test, y_train, y_test

# Load the diamonds dataset
diamonds_url = "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
diamonds = pd.read_csv(diamonds_url)

# Apply logarithmic transformation
diamonds['log_carat'] = np.log(diamonds['carat'])
diamonds['log_price'] = np.log(diamonds['price'])

# Prepare data for modeling
X = diamonds[['log_carat']].values
y = diamonds['log_price'].values

# Using simple_train_test_split function
X_train_simple, X_test_simple, y_train_simple, y_test_simple = simple_train_test_split(X, y, test_size=0.3)

# Using sklearn's train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize linear regression models
model_simple = LinearRegression()
model_sklearn = LinearRegression()

# Fit models
model_simple.fit(X_train_simple, y_train_simple)
model_sklearn.fit(X_train, y_train)

# Predictions
y_pred_simple = model_simple.predict(X_test_simple)
y_pred_sklearn = model_sklearn.predict(X_test)

# Calculate explained variance score
explained_variance_simple = explained_variance_score(y_test_simple, y_pred_simple)
explained_variance_sklearn = explained_variance_score(y_test, y_pred_sklearn)

# Print explained variance scores
print(f"Explained Variance Score (Simple Train-Test Split): {explained_variance_simple:.4f}")
print(f"Explained Variance Score (Sklearn Train-Test Split): {explained_variance_sklearn:.4f}")

# Visualization
plt.figure(figsize=(12, 6))

# Plotting results from simple_train_test_split
plt.subplot(1, 2, 1)
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Predicted (Simple Split)')
plt.title('Simple Train-Test Split')
plt.xlabel('Log Carat')
plt.ylabel('Log Price')
plt.legend()

# Plotting results from sklearn train_test_split
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_sklearn, color='green', linewidth=2, label='Predicted (Sklearn Split)')
plt.title('Sklearn Train-Test Split')
plt.xlabel('Log Carat')
plt.ylabel('Log Price')
plt.legend()

plt.tight_layout()
plt.show()
