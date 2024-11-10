# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split

# Ensure the figs directory exists
os.makedirs('figs', exist_ok=True)

# Load the dataset
df = pd.read_csv('data/diamonds.csv')

# Apply logarithmic transformation to price and carat
df['log_price'] = np.log(df['price'])
df['log_carat'] = np.log(df['carat'])

# Prepare the feature matrix X and target vector y
X = df[['log_carat']].values  # Feature matrix (as a 2D array)
y = df['log_price'].values    # Target vector

# Define the simple_train_test_split function
def simple_train_test_split(X, y, test_size=0.3):
    n_training_samples = int((1.0 - test_size) * X.shape[0])
    X_train = X[:n_training_samples, :]
    y_train = y[:n_training_samples]
    X_test = X[n_training_samples:, :]
    y_test = y[n_training_samples:]
    return X_train, X_test, y_train, y_test

# Split the data using the simple_train_test_split function
X_train_simple, X_test_simple, y_train_simple, y_test_simple = simple_train_test_split(X, y, test_size=0.3)

# Split the data using sklearn's train_test_split function
X_train_sklearn, X_test_sklearn, y_train_sklearn, y_test_sklearn = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create and train the linear regression model using simple_train_test_split data
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)

# Predict and evaluate using the simple_train_test_split test data
y_pred_simple = model_simple.predict(X_test_simple)
evs_simple = explained_variance_score(y_test_simple, y_pred_simple)
print(f'Explained Variance Score with simple_train_test_split: {evs_simple:.4f}')

# Create and train the linear regression model using sklearn's train_test_split data
model_sklearn = LinearRegression()
model_sklearn.fit(X_train_sklearn, y_train_sklearn)

# Predict and evaluate using sklearn's train_test_split test data
y_pred_sklearn = model_sklearn.predict(X_test_sklearn)
evs_sklearn = explained_variance_score(y_test_sklearn, y_pred_sklearn)
print(f'Explained Variance Score with sklearn\'s train_test_split: {evs_sklearn:.4f}')

# Visualization to compare predictions vs actual values for both methods
plt.figure(figsize=(12, 6))

# Plot for simple_train_test_split
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test_simple, y=y_pred_simple, alpha=0.5, color='blue')
plt.plot([y_test_simple.min(), y_test_simple.max()], [y_test_simple.min(), y_test_simple.max()], 'r--')
plt.title('Predicted vs Actual (Simple Split)')
plt.xlabel('Actual log(Price)')
plt.ylabel('Predicted log(Price)')

# Plot for sklearn's train_test_split
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test_sklearn, y=y_pred_sklearn, alpha=0.5, color='green')
plt.plot([y_test_sklearn.min(), y_test_sklearn.max()], [y_test_sklearn.min(), y_test_sklearn.max()], 'r--')
plt.title('Predicted vs Actual (Sklearn Split)')
plt.xlabel('Actual log(Price)')
plt.ylabel('Predicted log(Price)')

plt.tight_layout()
plt.savefig('figs/q3_predicted_vs_actual_comparison.png')
plt.show()
plt.close()

# Additional visualization of residuals
plt.figure(figsize=(12, 6))

# Residuals for simple_train_test_split
plt.subplot(1, 2, 1)
sns.histplot(y_test_simple - y_pred_simple, kde=True, color='blue')
plt.title('Residuals Distribution (Simple Split)')
plt.xlabel('Residuals')

# Residuals for sklearn's train_test_split
plt.subplot(1, 2, 2)
sns.histplot(y_test_sklearn - y_pred_sklearn, kde=True, color='green')
plt.title('Residuals Distribution (Sklearn Split)')
plt.xlabel('Residuals')

plt.tight_layout()
plt.savefig('figs/q3_residuals_comparison.png')
plt.show()
plt.close()
