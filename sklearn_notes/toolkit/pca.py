import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# 1. Load the Olivetti faces dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data  # shape: (n_samples, 4096)  -> 64x64 images
y = faces.target  # labels for the faces

# 2. Split into train and test sets (best practice: keep a portion for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# 3. Fit PCA on the training set
#    - n_components can be tuned; 150 is often enough to capture most variance
#    - whiten=True can sometimes help with better component scaling
pca = PCA(n_components=150, whiten=True, random_state=42)
pca.fit(X_train)

# 4. Transform both train and test sets into the reduced PCA space
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Original dimensionality:", X_train.shape[1])
print("Reduced dimensionality:", X_train_pca.shape[1])

# 5. Visualize the top 10 PCA components (often called eigenfaces)
#    Each component can be reshaped to the original 64x64 dimension
num_components_to_show = 10
eigenfaces = pca.components_[:num_components_to_show].reshape((num_components_to_show, 64, 64))

# Plot each eigenface in its own figure
for i, face_component in enumerate(eigenfaces, start=1):
    plt.figure()
    plt.title(f"PCA Component #{i}")
    plt.imshow(face_component, cmap="gray")
    plt.axis("off")
    plt.show()

# 6. Reconstruct a few random faces from the test set to show PCA performance
#    We transform the images into PCA space, then inverse transform
#    back to the original space (approximately).
num_faces_to_reconstruct = 5
random_indices = np.random.choice(len(X_test), size=num_faces_to_reconstruct, replace=False)

for idx in random_indices:
    original_face = X_test[idx]
    transformed_face = X_test_pca[idx]
    reconstructed_face = pca.inverse_transform(transformed_face)
    
    # Convert them back to 64x64 for visualization
    original_image = original_face.reshape(64, 64)
    reconstructed_image = reconstructed_face.reshape(64, 64)
    
    # Plot original face
    plt.figure()
    plt.title("Original Face")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Plot reconstructed face
    plt.figure()
    plt.title("Reconstructed Face")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")
    plt.show()