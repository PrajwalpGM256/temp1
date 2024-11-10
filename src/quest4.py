import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import explained_variance_score

# Load the diamonds dataset
diamonds_url = "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
diamonds = pd.read_csv(diamonds_url)

# Apply logarithmic transformation
diamonds['log_carat'] = np.log(diamonds['carat'])
diamonds['log_price'] = np.log(diamonds['price'])

# Select predictors and target
X = diamonds[['log_carat']]
y = diamonds['log_price']

# Categorical features for encoding
cat_features = ['cut', 'color', 'clarity']

# Initialize variables to track the best model
best_score = -np.inf
best_model = None
best_feature_names = None
best_feature_matrix_shape = None

# Iterate over each categorical feature as the second predictor
for feature_name in cat_features:
    # Prepare the feature matrix with one-hot encoding
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [feature_name])], remainder='passthrough')
    X_encoded = ct.fit_transform(diamonds[[feature_name, 'log_carat']])

    # Initialize linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_encoded, y)

    # Predictions
    y_pred = model.predict(X_encoded)

    # Calculate explained variance score
    explained_variance = explained_variance_score(y, y_pred)

    # Print results for each model
    print(f"Explained Variance Score (Log Carat + {feature_name}): {explained_variance:.4f}")

    # Track the best performing model
    if explained_variance > best_score:
        best_score = explained_variance
        best_model = model
        best_feature_names = [feature_name, 'log_carat']
        best_feature_matrix_shape = X_encoded.shape

# Report the best performing model
print(f"\nBest performing 2-input model: Log Carat + {best_feature_names} with Explained Variance Score: {best_score:.4f}")
print(f"Feature matrix shape of the best model: {best_feature_matrix_shape}")

