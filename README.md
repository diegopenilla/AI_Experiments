# ML Notes

##  **Ridge Regression: A Regularized Linear Regression Model**

Ridge Regression is a **regularized version of Linear Regression** that helps prevent **overfitting** by adding a **penalty** to large coefficients. It is particularly useful when dealing with **multicollinearity** (high correlation between independent variables).

Ridge Regression minimizes the following cost function:

```math
J(\mathbf{w}) = \sum_{i=1}^{N} (y_i - \mathbf{w}^T x_i)^2 + \lambda \sum_{j=1}^{p} w_j^2
```

where:
- $J(\mathbf{w})$ is the **loss function** (sum of squared errors with a penalty term).
- $y_i$ is the **actual output**.
- $\mathbf{w}^T x_i$ is the **predicted output**.
- $\lambda$(alpha) is the **regularization parameter**, controlling the penalty on large coefficients.
- $\sum w_j^2$ is the **L2 regularization term**, which discourages large values of weights.

### **1. Regularization (L2 Penalty)**
- The **L2 penalty** $\lambda \sum w_j^2$  **shrinks** the magnitude of coefficients, preventing them from becoming too large.
- Helps in cases where **features are highly correlated**, reducing overfitting.

### **2. Effect of the Regularization Parameter $\lambda$**

- **I 5\lambda = 05** → Ridge Regression behaves like **Ordinary Least Squares (OLS)**.
- **If $\lambda$  is large** → Model coefficients are heavily **penalized**, leading to smaller values and preventing overfitting.
- **Choosing $\lambda$** → Cross-validation is commonly used to find the optimal value.


___


## **Lasso Regression: A Regularized Linear Regression Model**


Lasso (Least Absolute Shrinkage and Selection Operator) Regression is a **regularized version of Linear Regression** that uses an **L1 penalty** to shrink some coefficients to exactly zero.

- This makes it useful for **feature selection** and reducing model complexity.


Lasso Regression minimizes the following cost function:

```math
J(\mathbf{w}) = \sum_{i=1}^{N} (y_i - \mathbf{w}^T x_i)^2 + \lambda \sum_{j=1}^{p} |w_j|
```

where:
- $J(\mathbf{w})$ is the **loss function** (sum of squared errors with an L1 penalty).
- $\mathbf{w}^T x_i$ is the **predicted output**.
- $\lambda$ is the **regularization parameter**, controlling the strength of penalty.
- $\sum |w_j|$ is the **L1 regularization term**, enforcing sparsity.

### **1. Regularization (L1 Penalty)**
- The **L1 penalty** \( \lambda \sum |w_j| \) encourages some coefficients to become **exactly zero**.
- This leads to **automatic feature selection**, as irrelevant features are eliminated.
- Helps in handling **high-dimensional datasets** where feature selection is necessary.

### **2. Effect of the Regularization Parameter \( \lambda \)**
- **If \( \lambda = 0 \)** → Lasso Regression behaves like **Ordinary Least Squares (OLS)**.
- **If \( \lambda \) is large** → More coefficients shrink to **zero**, making the model sparse.
- **Choosing \( \lambda \)** → Use **cross-validation** to find the optimal value.

### **Conclusion**
- **Lasso Regression** is ideal for **feature selection** and handling **sparse datasets**.
- The **L1 regularization** term leads to **simpler models** with fewer, more relevant features.
- **Tuning \( \lambda \)** is crucial to balancing model sparsity and predictive performance.

____ 

<br>

## K-Nearest Neighbors (KNN) Classifier

KNN is a **non-parametric, instance-based** algorithm that classifies a data point based on the majority class of its **k** nearest neighbors.

Given a dataset:

```math
D = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}
```

where $x_i$is a feature vector and $y_i$is the class label.  
For a test point $x$, KNN assigns a class: 

```math
\hat{y} = \arg\max_{c \in C} \sum_{i \in N_k(x)} \mathbb{1}(y_i = c)
```

where:
- C is the set of class labels.
- $N_k(x)$ is the set of k nearest neighbors of x based on distance. 
- $1 (y_i = c)$ is the indicator function, a counter that adds 1 if the neighbor belongs to the class, otherwise 0. 


### Definition of a Neighbor

A neighbor of a point  x  is any other data point in the dataset whose distance from  x  is among the  k  smallest distances. Each data point in the dataset consists of:

- A feature vector  $x_i$  (e.g., age, height, weight).
- A label  $y_i$  (for classification) or a numerical value  $y_i$  (for regression).
- When we receive a new test point  x , we:
	1.	Compute the distance from  x  to all data points in the training set.
	2.	Sort the distances in ascending order.
	3.	Select the  k  closest points as neighbors.


**Euclidean Distance** (default):

```math
d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}
```

Other options:
- **Manhattan Distance**:

  ```math
  d(x, x') = \sum_{i=1}^{n} |x_i - x'_i|
  ```

- **Minkowski Distance** (generalized):

  ```math
  d(x, x') = \left( \sum_{i=1}^{n} |x_i - x'_i|^p \right)^{\frac{1}{p}}
  ```

### Choosing \( k \)
- **Small \( k \)**: Sensitive to noise, may overfit.
- **Large \( k \)**: Smooths decision boundary, may underfit.
- Use **cross-validation** to find the best \( k \).

### Weighted Voting
Instead of simple majority voting, assign **weights** based on distance:

```math
w_i = \frac{1}{d(x, x_i) + \epsilon}
```

where $epsilon$ is a small constant to prevent division by zero.

## KNearestRegressor

Instead of voting, KNN regression averages the values of the nearest neighbors.

```math
\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y_i
```

#### Weighted KNN Regressor 

Instead of a simple average, this is a weighted average where closer neighbors have more influence.

```math
\hat{y} = \frac{\sum_{i \in N_k(x)} w_i y_i}{\sum_{i \in N_k(x)} w_i}
```
_____

<br>

## **Decision Trees: Supervised Learning**

Decision Trees are used for **classification and regression** by recursively splitting data into subsets based on feature values. The goal is to create **pure nodes** where all samples belong to the same category (classification) or have minimal variance (regression).

At each node (point where the data is split based on a features' value), the best split is chosen using an **impurity measure**:


### **Gini Impurity**
```math
G = 1 - \sum_{i=1}^{C} p_i^2
```
where:
- $p_i$ is the probability of class \( i \) in the node.
- Lower values mean purer splits.

### **Entropy (Information Gain)**
```math
H = -\sum_{i=1}^{C} p_i \log_2 p_i
```
A split is chosen to maximize **Information Gain**:
```math
IG = H_{parent} - \sum_{j=1}^{k} \frac{|D_j|}{|D|} H_j
```

###  Regression Splitting Criteria
For regression, Decision Trees minimize variance using **Mean Squared Error (MSE)**:
```math
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y})^2
```
where \( \hat{y} \) is the mean target value in the node.

### Stopping Conditions
The tree stops growing when:
- A **maximum depth** is reached.
- Nodes have fewer than a **minimum number of samples**.
- Further splits do not significantly reduce impurity.

### Overfitting and Pruning
- **Pre-pruning**: Limits tree depth or minimum samples per node.
- **Post-pruning**: Removes branches that do not improve generalization.

### Decision Trees: Classification vs. Regression
| Feature | Classification | Regression |
|---------|---------------|------------|
| **Output** | Class labels | Continuous values |
| **Splitting Metric** | Gini, Entropy | MSE |
| **Prediction Rule** | Majority class in leaf | Mean of leaf values |

### **Conclusion**
- Decision Trees are **easy to interpret** and **handle non-linearity well**.
- **Overfitting** is common, so **pruning** and **hyperparameter tuning** are crucial.
- They serve as the foundation for powerful models like **Random Forests** and **Gradient Boosting**.

____


<br>

## **Random Forests**
Random Forest is an **ensemble learning** algorithm that combines multiple **Decision Trees** to improve **accuracy, reduce overfitting**, and enhance **generalization**.

- **Bagging (Bootstrap Aggregation)**: Each tree is trained on a **random subset** of data sampled **with replacement**.
- **Random Feature Selection**: At each split, only a **random subset of features** is considered.
- **Majority Voting (Classification)**: The final class is chosen based on the **most common prediction** among trees.
- **Averaging (Regression)**: The final prediction is the **average of all tree outputs**.

---

### **1. Bagging - Bootstrap Aggregation**
Each tree is trained on a dataset $D^b$ drawn **randomly with replacement** from the original dataset \( D \).

```math
D^b = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}, \quad D^b \subseteq D
```

where $N$ is the number of training samples.

---

### **2️. Random Feature Selection**
At each split, only a **random subset** of features \( F_b \) is considered:

```math
F_b \subseteq F, \quad |F_b| < |F|
```

where $F$is the total set of features.

---

### **3️. Prediction - Aggregation of Trees**
For **classification**, the final prediction is the **majority vote** among trees:

```math
\hat{y} = \arg\max_c \sum_{b=1}^{B} \mathbb{1}(T_b(x) = c)
```
where:
- $B$ is the number of trees.
- $T_b(x)$ is the class prediction from the \( b^{th} \) tree.
- $c$ is a class label.

For **regression**, the final prediction is the **average of all tree outputs**:

```math
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
```

---

## **Advantages**
- **Reduces Overfitting**: Aggregating multiple trees lowers variance.  
- **Handles High-Dimensional Data**: Feature randomness helps in feature selection.  
- **Scales Well**: Parallelizable across multiple processors.  
- **Robust to Noise**: Reduces overfitting by using bootstrap sampling.


<br>

---

##  K-Means Clustering
K-Means is a centroid-based clustering algorithm that partitions a dataset into $K$ clusters by minimizing intra-cluster variance.

Given a dataset $X = \{x_1, x_2, \dots, x_n\}$, K-Means clustering seeks to partition it into $K$ disjoint clusters $C = \{C_1, C_2, \dots, C_K\}$ by solving the optimization problem:

```math
\arg \min_C \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
```

where:
- $\mu_k$ is the centroid (mean) of cluster $C_k$
- $\| x_i - \mu_k \|^2$ is the squared Euclidean distance between a data point and its assigned cluster centroid

#### Algorithm Workflow
1. Initialize $K$ centroids randomly.
2. Assign each data point to the nearest centroid.
3. Update centroids by computing the mean of assigned points.
4. Repeat steps 2-3 until centroids converge or a maximum number of iterations is reached.

#### Advantages
- Computationally efficient for large datasets.
- Works well when clusters are well-separated and spherical.

#### Limitations
- Requires specifying $K$ beforehand.
- Sensitive to outliers and initial centroid placement.

---

### Agglomerative Clustering
Agglomerative clustering is a hierarchical, bottom-up approach that merges data points into clusters iteratively.

Given a dataset $X = \{x_1, x_2, \dots, x_n\}$, Agglomerative clustering starts with each point as its own cluster and merges clusters based on a distance metric $d(A, B)$. The objective is:

```math
\arg \min_{A, B} d(A, B)
```

where $A$ and $B$ are clusters, and $d(A, B)$ is the linkage criterion:
- **Single Linkage:**
  ```math
  d(A, B) = \min_{x_i \in A, x_j \in B} \| x_i - x_j \|
  ```
  (distance between the closest points in clusters)
- **Complete Linkage:**
  ```math
  d(A, B) = \max_{x_i \in A, x_j \in B} \| x_i - x_j \|
  ```
  (distance between the farthest points in clusters)
- **Average Linkage:**
  ```math
  d(A, B) = \frac{1}{|A||B|} \sum_{x_i \in A} \sum_{x_j \in B} \| x_i - x_j \|
  ```
  (average distance between all pairs in clusters)

#### Algorithm Workflow
1. Treat each data point as its own cluster.
2. Compute pairwise distances between all clusters.
3. Merge the closest clusters based on linkage criterion.
4. Repeat until a single cluster remains or a predefined number of clusters is reached.

#### Advantages
- Does not require specifying the number of clusters in advance.
- Can capture complex cluster structures.

#### Limitations
- Computationally expensive for large datasets ($O(n^2)$ complexity)

<br>

___

## Handling Imbalanced Datasets 

An imbalanced dataset occurs when one class significantly outnumbers the other(s), which can lead to biased model performance. For example, in fraud detection, the number of fraudulent transactions is much lower than legitimate ones.

## **Challenges of Imbalanced Data**
- **Bias towards majority class:** Models may predict the majority class more often, ignoring the minority class.
- **Poor generalization:** The model may not learn meaningful patterns for minority class instances.
- **Skewed performance metrics:** Accuracy may be misleading, as high accuracy can be achieved by always predicting the majority class.

## **Techniques to Handle Imbalanced Data**
### **Resampling Methods**
#### **Oversampling (Increase Minority Samples)**
- **Random Oversampling:** Duplicates minority class samples.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Generates synthetic data points.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

#### **Undersampling (Reduce Majority Samples)**
- **Random Undersampling:** Removes majority class samples.
- **Cluster-Based Undersampling:** Selects representative samples from the majority class.

```python
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler()
X_resampled, y_resampled = undersample.fit_resample(X, y)
```

### **3.2 Algorithm-Based Approaches**
- **Class Weighting:** Assigns higher weights to the minority class.

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train, y_train)
```


## Anomaly Detection

Anomaly detection aims to identify rare or unusual patterns that do not conform to expected behavior in a dataset.


### **1.1. Probability-Based Approach (Gaussian Distribution)**
For normally distributed data, anomalies can be detected using the probability density function (PDF):

```math
P(X) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(X - \mu)^2}{2\sigma^2}}
```
- $X$: Data point
- $\mu$: Mean of distribution
- $\sigma$: Standard deviation
- Anomalies occur when $P(X)$ is below a certain threshold (e.g., 3 standard deviations from the mean).

### **1.2. Distance-Based Approach (k-Nearest Neighbors, kNN)**
Anomalies can be detected based on distance from neighbors:

```math
D(X) = \frac{1}{k} \sum_{i=1}^{k} d(X, X_i)
```
- $D(X)$: Anomaly score of $X$
- $X_i$: k-nearest neighbors
- $d(X, X_i)$: Distance metric (e.g., Euclidean)
- Higher distance values indicate anomalies.

### **1.3. Isolation Forest**
Isolation Forest isolates anomalies based on tree partitioning. The anomaly score is:

```math
S(X, n) = 2^{- \frac{E(h(X))}{c(n)}}
```
- $E(h(X))$: Average path length of $X$ in the tree
- $c(n)$: Normalization factor
- Shorter path length means $X$ is more likely an anomaly.


### **Summary**
| Method | Approach | Suitable for |
|--------|---------|--------------|
| Gaussian Model | Probability Density | Normally distributed data |
| kNN | Distance-Based | Data with clusters |
| Isolation Forest | Tree Partitioning | High-dimensional data |

Selecting the right method depends on the dataset's distribution and the type of anomalies present.

____


## Confusion Matrix

A Confusion Matrix is a tabular representation of a classification model’s performance.


```math
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{TN} + \text{FN}}
```

```math
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
```

```math
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
````

```math
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

The F1 Score is the **harmonic mean** of Precision and Recall, providing a **balanced evaluation** of the classifier's performance.

## Receiver Operating Characteristic (ROC) Curve

The **ROC Curve** evaluates the performance of a binary classifier across different decision thresholds. It plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)**.


Given a classifier outputting predicted probabilities $\hat{y}_i$ for the positive class, the TPR and FPR at a threshold $\tau$ are defined as:

```math
\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}
```
```math
\text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}
```

The Area Under the ROC Curve (AUC-ROC) quantifies the classifier's performance across all thresholds. A higher AUC-ROC indicates better performance.

```math
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x))dx
```



## Cross Validation

Cross Validation (CV) is a technique to estimate the generalization performance of a machine learning model by partitioning the dataset into multiple subsets for training and validation.

### $K$-Fold Cross Validation

1. **Partition** the dataset $(x_i,y_i)_{i=1}^n$ into $K$ disjoint subsets (folds) of approximately equal size:
   $$
   \mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_K
   $$

2. **Iterate** over each fold $k\in\{1,\dots,K\}$:
   - Let $\mathcal{D}_k$ be the validation set, and let $\mathcal{D}_{-k}=\mathcal{D}\setminus\mathcal{D}_k$ be the training set.
   - Train the model $f_{\theta}^{(k)}$ on $\mathcal{D}_{-k}$.
   - Compute the validation loss:
     $$
     \ell_k=\frac{1}{|\mathcal{D}_k|} \sum_{(x_i,y_i) \in \mathcal{D}_k} L(f_{\theta}^{(k)}(x_i),y_i)
     $$

3. **Compute the final cross-validation score** by averaging over all $K$ folds:
   $$
   \ell_{\text{CV}}=\frac{1}{K} \sum_{k=1}^{K} \ell_k
   $$

### Leave-One-Out Cross Validation (LOOCV)

- A special case where $K=n$, meaning each data point is used once as a validation set while the rest serve as the training set.

### Why Use Cross Validation?

- Provides a more **reliable estimate** of model performance than a single train/test split.
- Reduces variance in performance estimation by averaging multiple training-validation runs.
- Useful for **hyperparameter tuning** when combined with techniques like **Grid Search**.
________

## Automatic Feature Selection

Feature selection is the process of selecting the most relevant features from a dataset to improve model performance, reduce overfitting, and enhance interpretability. Automatic feature selection methods help streamline this process by leveraging statistical and machine learning techniques.

### 1. Filter Methods
Filter methods evaluate features independently of the model by assessing their relationship with the target variable.

#### Mutual Information
Mutual information measures the dependence between a feature $X$ and the target variable $Y$:

```math
I(X, Y) = \sum_{x \in X} \sum_{y \in Y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)}
```

Where:
- $P(x,y)$: The joint probability of $X=x$ and $Y=y$, i.e., the probability that both the feature and target take specific values togethe
- $P(x)$: The marginal probability of $X=x$, representing how often a feature value appears in the dataset.
- 	$P(y)$: The marginal probability of $Y=y$, representing how often a target value appears.

A higher mutual information score indicates a stronger relationship between the feature and the target.

#### Correlation Coefficient
Measures the linear relationship between a feature $X$ and the target $Y$:

```math
\rho_{X, Y} = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}
```

Features with high absolute correlation are often retained, while highly correlated redundant features may be removed.

---

### 2. Wrapper Methods
Wrapper methods select features by training a model and evaluating performance using subsets of features.

#### Recursive Feature Elimination (RFE)
RFE iteratively removes the least important features by training a model and ranking feature importance:

1. Train a model with all features.
2. Compute feature importance scores.
3. Remove the least important feature(s).
4. Repeat until the desired number of features remains.

The optimal subset minimizes the validation loss:

```math
\theta^* = \arg\min_{\theta} \sum_{i=1}^{n} L(f_\theta(X_i), Y_i)
```

---

### 3. Embedded Methods
Embedded methods perform feature selection during model training by incorporating regularization techniques.

#### LASSO (L1 Regularization)
LASSO regression adds an $L_1$ penalty term to shrink less important feature coefficients to zero:

```math
\min_{\theta} \sum_{i=1}^{n} (y_i - f_\theta(X_i))^2 + \lambda \sum_{j=1}^{m} |\theta_j|
```

where $\lambda$ controls the strength of regularization. Higher $\lambda$ values lead to more aggressive feature selection.

#### Tree-Based Methods
Decision trees and ensemble models (e.g., Random Forest, XGBoost) naturally rank feature importance by evaluating the decrease in impurity caused by each feature.

The importance score for feature $X_j$ is computed as:

```math
I(X_j) = \sum_{t \in T} \Delta \, 	ext{Impurity}_t \cdot \, I(t, X_j)
```

where $T$ is the set of tree nodes, and $I(t, X_j)$ indicates whether feature $X_j$ was used for splitting at node $t$.

---

### 4. Stability and Performance Considerations
When applying feature selection, it is essential to:
- Use **cross-validation** to prevent overfitting:

```math
\ell_{\text{CV}} = \frac{1}{K} \sum_{k=1}^{K} \ell_k
```

- Compare different feature selection methods and validate the final subset on a separate test set.
- Consider domain knowledge to avoid discarding useful but weakly correlated features.

Feature selection improves computational efficiency, enhances model interpretability, and can lead to better generalization performance.


<br>

## Principal Component Analysis (PCA) 

Principal Component Analysis is a dimensionality reduction technique that transforms high-dimensional data into fewer dimensions while preserving as much important information as possible.

## Intuition - Finding the Best View
Imagine taking a photo of a three-dimensional object:
- From some angles, it looks cluttered and unclear.
- But if you rotate the object, you can find the best angle that captures the most details in two dimensions.

PCA does the same thing. It finds the best way to project high-dimensional data into a lower dimension while keeping important patterns.

## Core Concepts
### Principal Components:
- PCA finds new axes (directions), called Principal Components, that capture the most variance (spread of data).
- The first Principal Component (PC1) is the direction where the data varies the most.
- The second Principal Component (PC2) is perpendicular to the first and captures the next highest variance.

### Eigenvectors & Eigenvalues
- **Eigenvectors** represent the directions of the new axes (principal components).
- **Eigenvalues** measure the importance (variance captured) by each principal component.

More variance means more information retained.

### Dimensionality Reduction
- We keep only the top $k$ principal components that explain most of the variance.
- This helps reduce noise, speed up computations, and avoid overfitting.

## Formulation
Given a dataset $X$ with $n$ features, PCA follows these steps:

1. Standardize the data (zero mean, unit variance).
2. Compute the covariance matrix $\Sigma$:

   ```math
   \Sigma = \frac{1}{n} X^T X
   ```

3. Find the eigenvectors and eigenvalues of $\Sigma$.
4. Sort eigenvectors by eigenvalues (largest to smallest).
5. Select the top $k$ eigenvectors to form a transformation matrix $W$.
6. Project the data onto new axes:

   ```math
   X' = X W
   ```

## When to Use PCA?
- When you have high-dimensional data and need to reduce complexity.
- When you want to remove noise and improve efficiency.
- When you want to visualize data in two or three dimensions.

## Key Trade-offs
- PCA loses some information (variance).
- PCA assumes linear relationships (not ideal for highly nonlinear data).

See PCA in Action, the code below demonstrates PCA on a sample face image dataset. The principal components are visualized to show the most important directions in the data.

```bash
python3 ./toolkit/pca.py
```




<br>

____________________

## Grid Search

Grid Search is a systematic procedure to find the optimal hyperparameters for a machine learning model. Suppose we have:

- A dataset $(x_i,y_i)_{i=1}^n$, where each $x_i$ represents the features and $y_i$ the corresponding label or target.
- A set of possible hyperparameter values $\Lambda=\{\lambda_1,\lambda_2,\dots,\lambda_m\}$. Each $\lambda_j$ might be a single hyperparameter (e.g., regularization parameter) or a combination of multiple hyperparameters (e.g., learning rate, number of trees, etc.).

For each hyperparameter combination $\lambda_j\in\Lambda$:

1. Train the model $\theta_j$ using $\lambda_j$.
2. Evaluate the model performance using an error or score function $L(f_{\theta_j})$.
3. Select the $\lambda_j$ that yields the best (lowest or highest, depending on whether it is a loss or a score) average performance on the validation set.


__________


## One-Hot Encoding

One-Hot Encoding is a method to represent categorical variables as binary vectors, ensuring that machine learning models can process them numerically without imposing an ordinal relationship.


Given a categorical variable $X$ with $m$ unique categories:
$$
X = \{x_1, x_2, \dots, x_m\}
$$

Each category $x_i$ is transformed into a binary vector of length $m$:

$$
\text{OneHot}(x_i) = [b_1, b_2, \dots, b_m]
$$

where:

$$
b_j =
\begin{cases}
1, & \text{if } x_i = x_j \\
0, & \text{otherwise}
\end{cases}
$$

### Key Benefits

- Eliminates ordinal relationships in categorical data.
- Ensures compatibility with machine learning models.
- Can increase dimensionality, so alternatives like **target encoding** or **embeddings** may be preferable for high-cardinality data.

<br>

____ 
# Scikit-Learn Pipeline

A **Pipeline** in Scikit-Learn is a structured way to automate a machine learning workflow by chaining together multiple processing steps, ensuring a streamlined and reproducible approach to model training and evaluation.


A Pipeline consists of a sequence of transformations $T_1, T_2, ..., T_k$ followed by a final estimator $f_{\theta}$. Given an input dataset $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$, the transformations and model training can be defined as:

```math
z_i=T_k(...(T_2(T_1(x_i)))) \quad \forall i \in \{1,2,\dots,n\}
```
where each $T_j$ represents a transformation (e.g., scaling, encoding, PCA), and the final estimator $f_{\theta}$ is trained on the transformed dataset:

```math
\theta^* = \arg\min_{\theta} \sum_{i=1}^{n} L(f_{\theta}(z_i),y_i)
```
where $L(\cdot)$ is a loss function measuring prediction error.

In Python e.g:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize features
    ('pca', PCA(n_components=5)),  # Step 2: Reduce dimensionality
    ('classifier', LogisticRegression())  # Step 3: Train model
])

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

```

This provides  an efficient way to structure ML workflows, ensuring consistency and ease of use. By encapsulating preprocessing and modeling into a single object, they prevent data leakage and simplify hyperparameter tuning, making them an essential tool in modern machine learning.



##  Scalers in Scikit-Learn

Feature scaling is essential in machine learning to ensure all features contribute equally to model training. Scikit-Learn provides various scalers:

## 1. **StandardScaler**
Standardizes features by removing the mean and scaling to unit variance:

```math
X' = \frac{X - \mu}{\sigma}
```
- $\mu$: Mean of feature
- $\sigma$: Standard deviation  
- Suitable for normally distributed data.

## 2. **MinMaxScaler**
Scales features to a fixed range [0,1]:

```math
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
```
- Retains original data distribution.
- Sensitive to outliers.

## 3. **RobustScaler**
Uses median and IQR (Interquartile Range) to reduce the impact of outliers:

```math
X' = \frac{X - \text{median}(X)}{\text{IQR}(X)}
```
- IQR = $Q_3 - Q_1$ (75th - 25th percentile)
- More robust to outliers than StandardScaler.

## 4. **Normalizer**
Scales each sample (row) to unit norm:

```math
X' = \frac{X}{\|X\|}
```
- Useful for text classification and sparse data.
- Ensures all samples have the same magnitude.

### **Summary**
| Scaler | Effect | Outlier Sensitivity |
|--------|--------|---------------------|
| StandardScaler | Zero mean, unit variance | High |
| MinMaxScaler | Scale to [0,1] | High |
| RobustScaler | Median & IQR scaling | Low |
| Normalizer | Normalize row-wise | N/A |

Choose the scaler based on data characteristics and sensitivity to outliers.


## Imputers in Scikit-Learn

Handling missing values is crucial in machine learning. Scikit-Learn provides various imputers to fill missing values effectively.

## 1. **SimpleImputer**
Fills missing values using a specified strategy:

```math
X' = 
\begin{cases} 
X, & \text{if not missing} \\
\text{strategy}(X), & \text{if missing}
\end{cases}
```
- Strategies: `mean`, `median`, `most_frequent`, `constant`
- Suitable for numerical and categorical data.

## 2. **KNNImputer**
Uses the k-nearest neighbors to impute missing values:

```math
X'_i = \frac{1}{k} \sum_{j \in N(i)} X_j
```
- $N(i)$: k-nearest neighbors of sample $i$.
- Useful when missing values depend on nearby points.

## 3. **IterativeImputer**
Predicts missing values using regression models:

```math
X'_i = f(X_{\text{known}})
```
- Iteratively estimates missing values based on other features.
- Suitable for complex relationships in data.

## 4. **MissingIndicator**
Identifies missing values as a separate binary feature:

```math
M_i = \begin{cases} 1, & \text{if } X_i \text{ is missing} \\ 0, & \text{otherwise} \end{cases}
```
- Helps models learn patterns in missing data.

### **Summary**
| Imputer | Method | Suitable For |
|---------|--------|--------------|
| SimpleImputer | Mean/Median/Mode | Basic missing data |
| KNNImputer | Nearest neighbor averaging | Data with local dependencies |
| IterativeImputer | Predictive modeling | Complex feature relationships |
| MissingIndicator | Binary indicator | Feature engineering |

Choose the imputer based on data characteristics and the nature of missing values.


## MLOps

MLOps (Machine Learning Operations) is the practice of automating and streamlining the lifecycle of machine learning models including:
-  development
-  deployment 
-  monitoring, 
-  and maintenance. 

It integrates principles from DevOps and applies them to ML workflows.

### **1.  Model Training as an Optimization Problem**
The goal of training a machine learning model is to minimize a loss function $L(\theta)$ where $\theta$ represents the model parameters:

```math
\theta^* = \arg\min_{\theta} L(\theta)
```

where:
- $L(\theta)$: Loss function (e.g., MSE, Cross-Entropy)
- $\theta$: Model parameters (weights, biases)

### **2. Model Deployment as a Function Approximation**
Once trained, an ML model is a function $f(X, \theta)$ that maps input features $X$ to predictions $Y$:

```math
Y = f(X, \theta^*)
```

where $\theta^*$ are the optimized parameters.

### **2.3. Model Monitoring using Performance Metrics**
Deployed models must be monitored over time using metrics such as accuracy, precision, recall, and drift detection:

```math
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
```

**Drift detection** measures changes in data distribution:

```math
D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}
```

where:
- $D_{KL}$: Kullback-Leibler divergence
- $P(x)$: Distribution of training data
- $Q(x)$: Distribution of incoming data

---

---

### **3.1. Model Training and Hyperparameter Tuning**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load and split dataset
data = pd.read_csv("your_dataset.csv")
X = data.drop(columns=["target"])
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {"classifier__n_estimators": [50, 100, 200]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Save the best model
joblib.dump(grid_search.best_estimator_, "best_model.pkl")
```

### **3.2. Model Deployment & Monitoring**
```python
# Load model and make predictions
model = joblib.load("best_model.pkl")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Monitor Data Drift (Basic Example)
def detect_drift(X_train, X_test):
    mean_diff = abs(X_train.mean() - X_test.mean())
    print("Mean Differences:", mean_diff)

detect_drift(X_train, X_test)
```

| Stage | Process |
|-------|---------|
| Model Training | Train and tune the ML model |
| Model Deployment | Save and load the model for inference |
| Model Monitoring | Track model performance and detect data drift |

MLOps ensures that machine learning models remain reliable, scalable, and maintainable throughout their lifecycle.
