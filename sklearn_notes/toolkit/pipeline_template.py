from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Load dataset (replace with actual dataset)
data = pd.read_csv("your_dataset.csv")

# Define feature columns
numerical_features = ["num_feature_1", "num_feature_2"]  # Numerical columns
categorical_features = ["cat_feature_1", "cat_feature_2"]  # Categorical columns

target = "target_column"

# Split dataset
X = data[numerical_features + categorical_features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical features (impute + scale)
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Define preprocessing for categorical features (impute + one-hot encode)
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Define full pipeline with preprocessing and model
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid for tuning
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
import joblib
joblib.dump(best_model, "best_ml_pipeline.pkl")

# Load and use the model later
loaded_model = joblib.load("best_ml_pipeline.pkl")
y_new_pred = loaded_model.predict(X_test)
