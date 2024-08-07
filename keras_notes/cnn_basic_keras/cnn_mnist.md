
# 🧠 Convolutional Neural Networks Foundations - Keras Implementation

Convolutional Neural Networks (CNNs) process image data through specialized layers designed to capture spatial patterns.

## Network Architecture

### **Convolutional Layers**
The first transformation applies filters to detect local patterns:
```math
\text{Conv2D}(X) = \text{ReLU}(W * X + b)
```
Where:
- $W$ represents learnable filters
- $*$ denotes the convolution operation
- $\text{ReLU}$ applies element-wise non-linearity
- $b$ is the bias term

Implementation uses:
- First layer: 32 filters with 3×3 kernels
- Second layer: 64 filters with 3×3 kernels
- Convolution operation computes the dot product between the filter W and local regions of the input X

### **Pooling Layers**
Pooling reduces spatial dimensions while preserving important features:
```math
\text{MaxPool}(X)_{i,j} = \max_{m,n \in R_{i,j}} X_{m,n}
```
Where $R_{i,j}$ represents the pooling region.

Our implementation uses:
- 2×2 max pooling after each convolutional layer, reducing dimensions by half

### **Flattening**
Converts feature maps to vectors for dense layer processing:
```math
\text{Flatten}(X) = [x_1, x_2, ..., x_n]
```

### **Dropout Regularization**
During training, randomly disables neurons to prevent overfitting:
```math
\text{Dropout}(X, p) = X \odot \text{Bernoulli}(1-p)
```
Where $p=0.5$ is the dropout probability.

### **Dense Output Layer**
The final layer produces class probabilities:
```math
\text{Dense}(X) = \text{Softmax}(WX + b)
```
Where Softmax normalizes outputs into a probability distribution:
```math
\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
```

## Training Process

### **Loss Function: Categorical Cross-Entropy**
```math
L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} y_{i,j} \log(\hat{y}_{i,j})
```
Where:
- $y_{i,j}$ is the true label (one-hot encoded)
- $\hat{y}_{i,j}$ is the predicted probability

### **Optimization: Adam**
The Adam optimizer adapts learning rates for each parameter:
1. Calculates first and second moments of gradients
2. Applies bias correction
3. Updates parameters with adaptive step sizes

```math
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

### **Training Configuration**
- **Batch Size**: 128 samples per gradient update
- **Epochs**: 15 passes through the entire dataset
- **Validation Split**: 10% of training data for performance monitoring

## Model Evaluation

### **Accuracy Metric**
```math
\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}(\text{argmax}(\hat{y}_i) = \text{argmax}(y_i))
```

The model is evaluated on held-out test data to measure:
1. **Test Loss**: Final cross-entropy value on unseen examples
2. **Test Accuracy**: Percentage of correctly classified digits


## CNN vs. Traditional Neural Networks

CNNs offer significant advantages for image processing:

1. **Parameter Efficiency**: Weight sharing reduces parameters by ~98% compared to fully-connected networks
2. **Translation Invariance**: Detect features regardless of their position in the image
3. **Hierarchical Feature Learning**: Early layers detect edges, later layers capture complex patterns
4. **Spatial Context Preservation**: Maintains local relationships between pixels



## Data Preparation

### **Loading and Preprocessing MNIST**
```math
X_{normalized} = \frac{X}{255}
```

The pipeline transforms raw MNIST images through:
1. **Normalization**: Scaling pixel values from [0,255] to [0,1] range
2. **Channel Expansion**: Adding a channel dimension for compatibility with CNN layers
3. **One-Hot Encoding**: Converting integer labels to categorical vectors


## Execution

```python3
python3 cnn_mnist.py
```


**Input Layer**
- `keras.Input(shape=input_shape)` - Defines the input shape (28, 28, 1) for MNIST images
- Each image is 28×28 pixels with 1 channel (grayscale)

**First Convolutional Layer**
- `layers.Conv2D(32, kernel_size=(3, 3), activation="relu")`
- Applies 32 filters with 3×3 kernels to the input
- Each filter learns to detect a specific pattern (like edges or textures)
- ReLU activation introduces non-linearity
- Shape changes from (28, 28, 1) → (26, 26, 32)
  - Dimensions shrink by 2 in height/width due to the 3×3 kernel without padding
  - Depth increases from 1 to 32 (number of filters)

**First Pooling Layer**
- `layers.MaxPooling2D(pool_size=(2, 2))`
- Downsamples by taking maximum value in each 2×2 window
- Reduces spatial dimensions while preserving important features
- Shape changes from (26, 26, 32) → (13, 13, 32)
  - Dimensions halve in height/width
  - Depth remains unchanged

**Second Convolutional Layer**
- `layers.Conv2D(64, kernel_size=(3, 3), activation="relu")`
- Applies 64 filters with 3×3 kernels
- Captures more complex patterns by combining features from previous layer
- Shape changes from (13, 13, 32) → (11, 11, 64)
  - Dimensions shrink by 2 again
  - Depth increases from 32 to 64

**Second Pooling Layer**
- `layers.MaxPooling2D(pool_size=(2, 2))`
- Further downsamples the feature maps
- Shape changes from (11, 11, 64) → (5, 5, 64)
  - Dimensions halve again
  - Depth remains at 64

**Flatten Layer**
- `layers.Flatten()`
- Converts the 3D feature maps into a 1D vector
- Shape changes from (5, 5, 64) → (1600)
  - 5×5×64 = 1600 features

**Dropout Layer**
- `layers.Dropout(0.5)`
- Randomly deactivates 50% of neurons during training
- Prevents overfitting by forcing the network to learn redundant representations
- Shape remains (1600) but half the values are zeroed during each training step

**Output Layer**
- `layers.Dense(num_classes, activation="softmax")`
- Fully connected layer with 10 neurons (one per digit)
- Softmax activation converts raw scores to probability distribution
- Shape changes from (1600) → (10)
  - Final output is 10 probabilities (one for each digit class)

This architecture progressively extracts features from simple to complex while reducing spatial dimensions, then uses these features to classify the digits.

```python
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 
Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 1600)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 1600)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 10)                  │          16,010 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 
Total params: 34,826 (136.04 KB)
Trainable params: 34,826 (136.04 KB)
Non-trainable params: 0 (0.00 B)
Epoch 1/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7684 - loss: 0.7493 - val_accuracy: 0.9772 - val_loss: 0.0827
Epoch 2/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9639 - loss: 0.1184 - val_accuracy: 0.9837 - val_loss: 0.0551
Epoch 3/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 6s 15ms/step - accuracy: 0.9730 - loss: 0.0844 - val_accuracy: 0.9858 - val_loss: 0.0478
Epoch 4/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9757 - loss: 0.0768 - val_accuracy: 0.9892 - val_loss: 0.0409
Epoch 5/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9810 - loss: 0.0606 - val_accuracy: 0.9903 - val_loss: 0.0367
Epoch 6/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 15ms/step - accuracy: 0.9826 - loss: 0.0579 - val_accuracy: 0.9885 - val_loss: 0.0383
Epoch 7/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9839 - loss: 0.0546 - val_accuracy: 0.9908 - val_loss: 0.0361
Epoch 8/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9855 - loss: 0.0444 - val_accuracy: 0.9897 - val_loss: 0.0344
Epoch 9/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9853 - loss: 0.0460 - val_accuracy: 0.9913 - val_loss: 0.0327
Epoch 10/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9869 - loss: 0.0414 - val_accuracy: 0.9917 - val_loss: 0.0315
Epoch 11/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9879 - loss: 0.0374 - val_accuracy: 0.9920 - val_loss: 0.0292
Epoch 12/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9870 - loss: 0.0380 - val_accuracy: 0.9913 - val_loss: 0.0308
Epoch 13/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9877 - loss: 0.0355 - val_accuracy: 0.9920 - val_loss: 0.0303
Epoch 14/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9895 - loss: 0.0344 - val_accuracy: 0.9918 - val_loss: 0.0311
Epoch 15/15
422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9891 - loss: 0.0322 - val_accuracy: 0.9925 - val_loss: 0.0288

Test loss: 0.0263
Test accuracy: 0.9907
```