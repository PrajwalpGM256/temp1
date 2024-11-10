# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
import os

# Ensure the figs directory exists
os.makedirs('figs', exist_ok=True)

# Load the dataset
df = pd.read_csv('data/diamonds.csv')

# Apply logarithmic transformation to price and carat
df['log_price'] = np.log(df['price'])
df['log_carat'] = np.log(df['carat'])

# Prepare the target variable
y = df['log_price'].values  # Convert to NumPy array

# List of categorical variables to test
categorical_vars = ['cut', 'color', 'clarity']

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Univariate model from Question 2 for comparison
X_univariate = df[['log_carat']].values
model_univariate = LinearRegression()

# Lists to store scores
univariate_train_scores = []
univariate_test_scores = []

# Perform cross-validation for univariate model
for train_index, test_index in kf.split(X_univariate):
    X_train, X_test = X_univariate[train_index], X_univariate[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model_univariate.fit(X_train, y_train)
    y_train_pred = model_univariate.predict(X_train)
    y_test_pred = model_univariate.predict(X_test)
    
    train_score = explained_variance_score(y_train, y_train_pred)
    test_score = explained_variance_score(y_test, y_test_pred)
    
    univariate_train_scores.append(train_score)
    univariate_test_scores.append(test_score)

# Report univariate model scores
print("Univariate Model (log_carat):")
print(f"Train Explained Variance: {np.mean(univariate_train_scores):.4f} ± {np.std(univariate_train_scores):.4f}")
print(f"Test Explained Variance: {np.mean(univariate_test_scores):.4f} ± {np.std(univariate_test_scores):.4f}\n")

# Dictionary to store results for each model
results = {}

# Iterate over categorical variables
for var in categorical_vars:
    # One-hot encode the categorical variable
    df_encoded = pd.get_dummies(df[var], prefix=var)
    
    # Combine log_carat and the encoded variable
    X = pd.concat([df['log_carat'], df_encoded], axis=1).values  # Convert to NumPy array
    
    # Initialize lists to store scores
    train_scores = []
    test_scores = []
    
    # Perform cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict on training and testing data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate explained variance scores
        train_score = explained_variance_score(y_train, y_train_pred)
        test_score = explained_variance_score(y_test, y_test_pred)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Store the results
    results[var] = {
        'train_mean': np.mean(train_scores),
        'train_std': np.std(train_scores),
        'test_mean': np.mean(test_scores),
        'test_std': np.std(test_scores),
        'X_shape': X.shape
    }
    
    # Report scores for this model
    print(f'Model with "log_carat" and "{var}":')
    print(f"Train Explained Variance: {results[var]['train_mean']:.4f} ± {results[var]['train_std']:.4f}")
    print(f"Test Explained Variance: {results[var]['test_mean']:.4f} ± {results[var]['test_std']:.4f}\n")

# Identify the best model based on average test score
best_var = max(results, key=lambda x: results[x]['test_mean'])
best_result = results[best_var]

print("Best 2-input model using cross-validation:")
print(f'Variable added: {best_var}')
print(f"Train Explained Variance: {best_result['train_mean']:.4f} ± {best_result['train_std']:.4f}")
print(f"Test Explained Variance: {best_result['test_mean']:.4f} ± {best_result['test_std']:.4f}")
print(f'Feature matrix shape: {best_result["X_shape"]}')
