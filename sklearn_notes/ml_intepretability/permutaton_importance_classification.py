import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Load dataset
data = pd.read_csv('./datasets/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert to binary classification

# Select numerical features
feature_names = [col for col in data.columns if data[col].dtype in [np.int64]]
X = data[feature_names]

# Split dataset
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(train_X, train_y)

# Compute permutation importance
perm_importance = permutation_importance(model, val_X, val_y, random_state=1)

# Convert results to DataFrame
importances_df = pd.DataFrame({
    "Feature": val_X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

# Display results
print(importances_df)

