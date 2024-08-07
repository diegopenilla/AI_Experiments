import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Step 1: Generate Synthetic Data (Normal Distribution)
n_samples = 300
X_normal = np.random.normal(loc=0, scale=1, size=(n_samples, 2))  # Mean=0, Std=1

# Step 2: Introduce Anomalies (Outliers)
n_outliers = 20
X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))

# Combine normal data and outliers
X = np.vstack((X_normal, X_outliers))

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply Anomaly Detection Methods
## 1. Isolation Forest
iso_forest = IsolationForest(contamination=0.06, random_state=42)
pred_iso = iso_forest.fit_predict(X_scaled)

## 2. Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20)
pred_lof = lof.fit_predict(X_scaled)

## 3. Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
scores_gmm = gmm.score_samples(X_scaled)
pred_gmm = (scores_gmm < np.percentile(scores_gmm, 6)).astype(int)  # Approx. 6% anomalies

# Step 4: Visualization
plt.figure(figsize=(12, 4))

## Isolation Forest Plot
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=pred_iso, cmap='coolwarm', edgecolors='k')
plt.title("Isolation Forest")

## Local Outlier Factor Plot
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=pred_lof, cmap='coolwarm', edgecolors='k')
plt.title("Local Outlier Factor")

## Gaussian Mixture Model Plot
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=pred_gmm, cmap='coolwarm', edgecolors='k')
plt.title("Gaussian Mixture Model")

plt.show()
