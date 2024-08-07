import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate synthetic dataset with missing values
np.random.seed(42)
X = pd.DataFrame(np.random.rand(1000, 5), columns=[f'feature_{i}' for i in range(5)])
y = np.random.randint(0, 2, 1000)  # Binary classification target

# Introduce missing values randomly
missing_mask = np.random.rand(*X.shape) < 0.1  # 10% missing values
X[missing_mask] = np.nan

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define multiple imputation methods in a pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Replace with 'median' or 'most_frequent' if needed
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train and evaluate the pipeline
pipeline.fit(X_train, y_train)
print("Pipeline trained successfully with mean imputation.")

# Alternative: KNN Imputation
knn_pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),  # Uses nearest neighbors to impute missing values
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train and evaluate KNN imputation pipeline
knn_pipeline.fit(X_train, y_train)
print("Pipeline trained successfully with KNN imputation.")