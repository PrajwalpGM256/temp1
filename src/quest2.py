import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

diamonds_url = "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
diamonds = pd.read_csv(diamonds_url)

# Log transformation of carat and price
diamonds['log_carat'] = np.log(diamonds['carat'])
diamonds['log_price'] = np.log(diamonds['price'])

# Plotting log price vs. log carat
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_carat', y='log_price', data=diamonds)
plt.title('Log Price vs. Log Carat')
plt.xlabel('Log Carat')
plt.ylabel('Log Price')
plt.savefig('figs/Q2_Log Price vs Log Carat')
plt.show()

# Create the model
X = diamonds[['log_carat']]
y = diamonds['log_price']

model = LinearRegression()
model.fit(X, y)

# Predictions and explained variance score
y_pred = model.predict(X)
explained_variance = explained_variance_score(y, y_pred)

print(f"Explained Variance Score (Univariate Log Transformation): {explained_variance:.4f}")

# Prepare data for modeling
X = diamonds[['log_carat']]
y = diamonds['log_price']

# Initialize linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Calculate explained variance score
explained_variance = explained_variance_score(y, y_pred)

print(f"Explained Variance Score (Log-Log Transformation): {explained_variance:.4f}")

