from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample categorical data
data = np.array([['red'], ['green'], ['blue'], ['red'], ['green']])

# Create OneHotEncoder using 'sparse_output' instead of 'sparse'
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the data to one-hot encode it
one_hot = encoder.fit_transform(data)

print("Original Data:")
print(data)
print("\nOne-Hot Encoded Data:")
print(one_hot)
print("\nCategories:")
print(encoder.categories_)
