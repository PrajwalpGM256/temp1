import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
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

# Initialize variables to store scores
train_scores = []
test_scores = []

# Iterate over each categorical feature as the second predictor
for feature_name in cat_features:
    # Prepare the feature matrix with one-hot encoding
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [feature_name])], remainder='passthrough')
    X_encoded = ct.fit_transform(diamonds[[feature_name, 'log_carat']])

    # Initialize linear regression model
    model = LinearRegression()

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='explained_variance')

    # Track train and test scores
    train_scores.append(cv_scores.mean())
    test_scores.append(cv_scores.std())

    # Print results for each model
    print(f"Cross-validation Explained Variance Scores (Log Carat + {feature_name}):")
    for fold_idx, score in enumerate(cv_scores):
        print(f"Fold {fold_idx + 1}: {score:.4f}")
    print(f"Average Train Score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print()

# Find the best performing model based on average train score
best_index = np.argmax(train_scores)
best_feature_name = cat_features[best_index]

# Report the best performing model
print(f"Best performing 2-input model: Log Carat + {best_feature_name}")
print(f"Average Train Score: {train_scores[best_index]:.4f} +/- {test_scores[best_index]:.4f}")
