#!/usr/bin/env python3
"""
Optimized Decision Tree Regression with 1D and Multi-Output Tasks.

This script:
- Fits a Decision Tree on a **1D regression task** (sine function).
- Demonstrates **multi-output regression** (circle function).
- Compares different tree depths to show **underfitting vs overfitting**.
- Uses **visualization** to highlight decision tree predictions.

Reference:
- Inspired by: https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


def generate_1d_data(n_samples=80, noise_factor=3, random_state=1):
    """
    Generate a 1D noisy dataset based on a sine function.

    Args:
        n_samples (int): Number of samples.
        noise_factor (float): Controls noise magnitude.
        random_state (int): Random seed for reproducibility.

    Returns:
        X (numpy array): Feature matrix.
        y (numpy array): Target values with noise.
    """
    rng = np.random.RandomState(random_state)
    X = np.sort(5 * rng.rand(n_samples, 1), axis=0)  # Features in range [0,5]
    y = np.sin(X).ravel()
    y[::5] += noise_factor * (0.5 - rng.rand(n_samples // 5))  # Add noise to every 5th sample
    return X, y


def train_decision_trees(X, y, depths=[2, 5]):
    """
    Train Decision Tree models with different depths.

    Args:
        X (numpy array): Feature matrix.
        y (numpy array): Target values.
        depths (list): List of `max_depth` values to compare.

    Returns:
        models (dict): Dictionary of trained models with keys as depths.
    """
    models = {}
    for depth in depths:
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X, y)
        models[depth] = tree
    return models


def plot_1d_results(X, y, models, X_test):
    """
    Plot predictions for Decision Trees with different depths.

    Args:
        X (numpy array): Training feature matrix.
        y (numpy array): Training target values.
        models (dict): Dictionary of trained models.
        X_test (numpy array): Test feature matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="Training Data")

    for depth, model in models.items():
        y_pred = model.predict(X_test)
        plt.plot(X_test, y_pred, label=f"max_depth={depth}", linewidth=2)

    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Decision Tree Regression (1D Task)")
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_multioutput_data(n_samples=100, noise_factor=0.5, random_state=1):
    """
    Generate a multi-output dataset simulating a noisy circle.

    Args:
        n_samples (int): Number of samples.
        noise_factor (float): Noise level.
        random_state (int): Random seed.

    Returns:
        X (numpy array): Feature matrix.
        y (numpy array): Multi-output target values.
    """
    rng = np.random.RandomState(random_state)
    X = np.sort(200 * rng.rand(n_samples, 1) - 100, axis=0)  # Features in range [-100,100]
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T  # Circle function
    y[::5, :] += noise_factor - rng.rand(n_samples // 5, 2)  # Add noise to every 5th sample
    return X, y


def plot_multioutput_results(X, y, models, X_test):
    """
    Plot multi-output regression predictions for different tree depths.

    Args:
        X (numpy array): Training feature matrix.
        y (numpy array): Training target values.
        models (dict): Dictionary of trained models.
        X_test (numpy array): Test feature matrix.
    """
    plt.figure(figsize=(8, 6))
    s = 25
    plt.scatter(y[:, 0], y[:, 1], c="yellow", s=s, edgecolor="black", label="Training Data")

    colors = ["cornflowerblue", "red", "blue"]
    for (depth, model), color in zip(models.items(), colors):
        y_pred = model.predict(X_test)
        plt.scatter(y_pred[:, 0], y_pred[:, 1], c=color, s=s, edgecolor="black", label=f"max_depth={depth}")

    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.xlabel("Target 1")
    plt.ylabel("Target 2")
    plt.title("Multi-Output Decision Tree Regression")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """Main function to run both 1D and Multi-Output Decision Tree Regression."""
    # 🔹 1D Regression Task
    print("\n🔹 Running 1D Regression Task...")
    X_1d, y_1d = generate_1d_data()
    X_test_1d = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]  # Test points
    models_1d = train_decision_trees(X_1d, y_1d, depths=[2, 5])  # Train models
    plot_1d_results(X_1d, y_1d, models_1d, X_test_1d)  # Visualize results

    # 🔹 Multi-Output Regression Task
    print("\n🔹 Running Multi-Output Regression Task...")
    X_multi, y_multi = generate_multioutput_data()
    X_test_multi = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]  # Test points
    models_multi = train_decision_trees(X_multi, y_multi, depths=[2, 5, 8])  # Train models
    plot_multioutput_results(X_multi, y_multi, models_multi, X_test_multi)  # Visualize results


if __name__ == "__main__":
    main()