# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import os

# Ensure the figs directory exists
os.makedirs('figs', exist_ok=True)

# Load the dataset
df = pd.read_csv('data/diamonds.csv')

# Apply logarithmic transformation to price and carat
df['log_price'] = np.log(df['price'])
df['log_carat'] = np.log(df['carat'])

# Prepare the target variable
y = df['log_price']

# List to store explained variance scores
scores = []

# List of categorical variables to test
categorical_vars = ['cut', 'color', 'clarity']

# Univariate model from Question 2 for comparison
X_univariate = df[['log_carat']]
model_univariate = LinearRegression()
model_univariate.fit(X_univariate, y)
y_pred_univariate = model_univariate.predict(X_univariate)
evs_univariate = explained_variance_score(y, y_pred_univariate)
print(f'Explained Variance Score for Univariate Model: {evs_univariate:.4f}')

# Iterate over categorical variables
for var in categorical_vars:
    # One-hot encode the categorical variable
    df_encoded = pd.get_dummies(df[var], prefix=var)
    
    # Combine log_carat and the encoded variable
    X = pd.concat([df['log_carat'], df_encoded], axis=1)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict on the training data
    y_pred = model.predict(X)
    
    # Calculate explained variance score
    evs = explained_variance_score(y, y_pred)
    print(f'Explained Variance Score with "log_carat" and "{var}": {evs:.4f}')
    
    # Store the results
    scores.append({'variable': var, 'evs': evs, 'X_shape': X.shape})
    
# Find the best model
best_model = max(scores, key=lambda x: x['evs'])
print("\nBest 2-input model:")
print(f'Variable added: {best_model["variable"]}')
print(f'Explained Variance Score: {best_model["evs"]:.4f}')
print(f'Feature matrix shape: {best_model["X_shape"]}')
