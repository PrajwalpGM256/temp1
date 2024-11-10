# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

# Ensure the figs directory exists
os.makedirs('figs', exist_ok=True)

# Load the dataset
df = pd.read_csv('data/diamonds.csv')

# Plot the original relationship between price and carat
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='carat', y='price', alpha=0.5)
plt.title('Price vs. Carat (Original Scale)')
plt.xlabel('Carat')
plt.ylabel('Price')

plt.savefig('figs/q2_price_vs_carat_original.png')
plt.show()
plt.close()

# Apply logarithmic transformation to price and carat
df['log_price'] = np.log(df['price'])
df['log_carat'] = np.log(df['carat'])

# Plot the transformed relationship between log(price) and log(carat)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='log_carat', y='log_price', alpha=0.5, color='orange')
plt.title('Log(Price) vs. Log(Carat)')
plt.xlabel('Log(Carat)')
plt.ylabel('Log(Price)')
plt.savefig('figs/q2_log_price_vs_log_carat.png')
plt.show()
plt.close()

# Prepare data for linear regression on original scale
X_original = df[['carat']]
y_original = df['price']

# Prepare data for linear regression on log-transformed scale
X_log = df[['log_carat']]
y_log = df['log_price']

# Create and train the linear regression model on original data
model_original = LinearRegression()
model_original.fit(X_original, y_original)

# Predict using the model on original data
y_pred_original = model_original.predict(X_original)

# Calculate explained variance score on original data
evs_original = explained_variance_score(y_original, y_pred_original)
print(f'Explained Variance Score on Original Data: {evs_original:.4f}')

# Create and train the linear regression model on log-transformed data
model_log = LinearRegression()
model_log.fit(X_log, y_log)

# Predict using the model on log-transformed data
y_pred_log = model_log.predict(X_log)

# Calculate explained variance score on log-transformed data
evs_log = explained_variance_score(y_log, y_pred_log)
print(f'Explained Variance Score on Log-Transformed Data: {evs_log:.4f}')

# Plot the regression line on original data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='carat', y='price', alpha=0.5)
plt.plot(df['carat'], y_pred_original, color='red', label='Linear Regression Fit')
plt.title('Linear Regression on Original Data')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.legend()
plt.savefig('figs/q2_linear_regression_original.png')
plt.show()
plt.close()

# Plot the regression line on log-transformed data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='log_carat', y='log_price', alpha=0.5, color='orange')
plt.plot(df['log_carat'], y_pred_log, color='red', label='Linear Regression Fit')
plt.title('Linear Regression on Log-Transformed Data')
plt.xlabel('Log(Carat)')
plt.ylabel('Log(Price)')
plt.legend()
plt.savefig('figs/q2_linear_regression_log_transformed.png')
plt.show()
plt.close()
