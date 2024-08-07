import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(42)
X = pd.DataFrame(np.random.rand(1000, 10), columns=[f'feature_{i}' for i in range(10)])
y = np.random.randint(0, 2, 1000)  # Binary classification target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection Methods

# 1. SelectKBest with Mutual Information
selector_univariate = SelectKBest(mutual_info_classif, k=5)
X_train_selected = selector_univariate.fit_transform(X_train_scaled, y_train)
selected_features_univariate = X.columns[selector_univariate.get_support()]

# 2. Recursive Feature Elimination (RFE) with Logistic Regression
selector_rfe = RFE(LogisticRegression(), n_features_to_select=5)
selector_rfe.fit(X_train_scaled, y_train)
selected_features_rfe = X.columns[selector_rfe.support_]

# 3. LASSO Regularization
lasso = LassoCV(cv=5)
lasso.fit(X_train_scaled, y_train)
selected_features_lasso = X.columns[lasso.coef_ != 0]

# 4. Feature Importance from Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
selected_features_rf = X.columns[np.argsort(rf.feature_importances_)[-5:]]

# Print selected features from each method
print("Selected features (Univariate Selection):", selected_features_univariate.tolist())
print("Selected features (RFE):", selected_features_rfe.tolist())
print("Selected features (LASSO):", selected_features_lasso.tolist())
print("Selected features (Random Forest):", selected_features_rf.tolist())