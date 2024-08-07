#!/usr/bin/env python3
"""
K-Nearest Neighbors (KNN) Classifier Implementation Demo

This script demonstrates how to implement a KNN classifier using scikit-learn.
It loads a dataset, preprocesses the data, trains a KNN model, and evaluates its performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def load_dataset(dataset_name='iris'):
    """
    Load a dataset from scikit-learn.
    
    Args:
        dataset_name: Name of the dataset to load ('iris' or 'digits')
        
    Returns:
        X: Feature data
        y: Target labels
        feature_names: Names of features
        target_names: Names of target classes
    """
    if dataset_name.lower() == 'iris':
        dataset = load_iris()
    elif dataset_name.lower() == 'digits':
        dataset = load_digits()
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose 'iris' or 'digits'.")
    
    return dataset.data, dataset.target, dataset.feature_names, dataset.target_names


def visualize_dataset(X, y, feature_names, target_names, dataset_name):
    """
    Create a visualization of the dataset.
    
    Args:
        X: Feature data
        y: Target labels
        feature_names: Names of features
        target_names: Names of target classes
        dataset_name: Name of the dataset
    """
    plt.figure(figsize=(12, 5))
    
    # For Iris dataset, plot the first two features
    if dataset_name.lower() == 'iris':
        plt.subplot(1, 2, 1)
        for target in np.unique(y):
            plt.scatter(X[y == target, 0], X[y == target, 1], 
                        label=target_names[target], alpha=0.7, s=50)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Iris Dataset - First Two Features')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for target in np.unique(y):
            plt.scatter(X[y == target, 2], X[y == target, 3], 
                        label=target_names[target], alpha=0.7, s=50)
        plt.xlabel(feature_names[2])
        plt.ylabel(feature_names[3])
        plt.title('Iris Dataset - Last Two Features')
        plt.legend()
    
    # For Digits dataset, show example digits
    elif dataset_name.lower() == 'digits':
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(X[i].reshape(8, 8), cmap='gray')
            ax.set_title(f"Label: {y[i]}")
            ax.axis('off')
        plt.suptitle('Example Digits from the Dataset')
    
    plt.tight_layout()
    plt.show()


def create_and_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    """
    Create, train, and evaluate a KNN classifier.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        n_neighbors: Number of neighbors for KNN
        
    Returns:
        knn: Trained KNN model
        accuracy: Model accuracy on test data
    """
    # Create a pipeline with preprocessing and KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalize features
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    
    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print evaluation results
    print(f"\nKNN Classifier (n_neighbors={n_neighbors}):")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, accuracy, y_pred


def find_optimal_k(X_train, X_test, y_train, y_test, k_range=range(1, 31)):
    """
    Find the optimal value of k using cross-validation.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        k_range: Range of k values to test
        
    Returns:
        optimal_k: Optimal value of k
        accuracies: List of accuracies for each k
    """
    # Create a pipeline for GridSearchCV
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    
    # Set up GridSearchCV
    param_grid = {'knn__n_neighbors': k_range}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and score
    optimal_k = grid_search.best_params_['knn__n_neighbors']
    best_score = grid_search.best_score_
    
    print(f"\nOptimal number of neighbors (k): {optimal_k}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")
    
    # Plot accuracy vs k
    plt.figure(figsize=(10, 6))
    accuracies = [grid_search.cv_results_['mean_test_score'][i] for i in range(len(k_range))]
    plt.plot(k_range, accuracies, marker='o', linestyle='-')
    plt.title('Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-validation Accuracy')
    plt.xticks(k_range[::2])  # Show every other k value for cleaner plot
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_k, accuracies


def visualize_confusion_matrix(y_true, y_pred, class_names):
    """
    Visualize the confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels to the plot
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
    """Main function to run the KNN demonstration."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Select dataset (change to 'digits' if you want to use the digits dataset)
    dataset_name = 'iris'
    
    print(f"KNN Classification Demo using the {dataset_name.capitalize()} dataset")
    print("=" * 60)
    
    # Load dataset
    X, y, feature_names, target_names = load_dataset(dataset_name)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(target_names)}")
    print(f"Classes: {target_names}")
    print(f"Number of samples per class: {[sum(y == i) for i in range(len(target_names))]}")
    
    # Visualize dataset
    visualize_dataset(X, y, feature_names, target_names, dataset_name)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Find optimal k
    optimal_k, _ = find_optimal_k(X_train, X_test, y_train, y_test)
    
    # Train and evaluate KNN with optimal k
    knn_model, accuracy, y_pred = create_and_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=optimal_k)
    
    # Visualize confusion matrix
    visualize_confusion_matrix(y_test, y_pred, target_names)
    
    print("\nKNN Classification Complete!")


if __name__ == "__main__":
    main()
