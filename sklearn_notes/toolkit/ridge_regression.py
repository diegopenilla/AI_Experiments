#!/usr/bin/env python3
"""
Optimized Ridge and Lasso Regression using Scikit-Learn.

This script:
- Loads the Diabetes dataset.
- Scales the features.
- Uses built-in cross-validation to find the optimal alpha (λ) for Ridge and Lasso.
- Evaluates both models using MSE, R², and MAE.
- Visualizes the results, including predictions and feature importance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
X, y = load_diabetes(return_X_y=True)

# Split into training and test sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define alpha (λ) search space
alpha_values = np.logspace(-4, 2, 100)  # Finer granularity

# Ridge Regression with built-in CV to find optimal alpha
ridge_cv = RidgeCV(alphas=alpha_values, store_cv_values=True)
ridge_cv.fit(X_train, y_train)
ridge_pred = ridge_cv.predict(X_test)

# Lasso Regression with built-in CV to find optimal alpha
lasso_cv = LassoCV(alphas=alpha_values, max_iter=10000, cv=10, n_jobs=-1)
lasso_cv.fit(X_train, y_train)
lasso_pred = lasso_cv.predict(X_test)

# Model evaluation function
def evaluate_model(model_name, model, y_true, y_pred):
    print(f"\n{model_name} Regression Results (Optimal α = {model.alpha_:.6f}):")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

# Print evaluation results
evaluate_model("Ridge", ridge_cv, y_test, ridge_pred)
evaluate_model("Lasso", lasso_cv, y_test, lasso_pred)

# Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, ridge_pred, alpha=0.7, label="Ridge Predictions", color='blue')
plt.scatter(y_test, lasso_pred, alpha=0.7, label="Lasso Predictions", color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", label="Perfect Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ridge vs Lasso Regression Predictions (Optimal α)")
plt.legend()
plt.grid(True)
plt.show()

# Feature importance for Lasso (L1 regularization forces some coefficients to zero)
plt.figure(figsize=(10, 6))
coefficients = lasso_cv.coef_
feature_names = load_diabetes().feature_names
sns.barplot(x=coefficients, y=feature_names, palette="coolwarm")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Lasso Feature Importance (Non-zero Coefficients)")
plt.grid(True, alpha=0.3)
plt.show()