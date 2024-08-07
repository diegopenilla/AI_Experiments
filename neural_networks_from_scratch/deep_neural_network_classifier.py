#!/usr/bin/env python3

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from scipy import ndimage

# Custom type aliases for clarity
NDArray = np.ndarray
Parameters = Dict[str, NDArray]
Cache = Tuple[Tuple[NDArray, NDArray, NDArray], NDArray]
Gradients = Dict[str, NDArray]

def initialize_parameters(n_x: int, n_h: int, n_y: int) -> Parameters:
    """
    Initialize parameters for a 2-layer neural network.
    
    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def initialize_parameters_deep(layer_dims: List[int]) -> Parameters:
    """
    Initialize parameters for an L-layer neural network.
    
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def sigmoid(Z: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Sigmoid activation function.
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z: NDArray) -> Tuple[NDArray, NDArray]:
    """
    ReLU activation function.
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of relu(z), same shape as Z
    cache -- returns Z, useful during backpropagation
    """
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def linear_forward(A: NDArray, W: NDArray, b: NDArray) -> Tuple[NDArray, Tuple[NDArray, NDArray, NDArray]]:
    """
    Implement the linear part of a layer's forward propagation.
    
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    
    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for backward propagation
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(
    A_prev: NDArray,
    W: NDArray,
    b: NDArray,
    activation: str
) -> Tuple[NDArray, Cache]:
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache"; stored for backward propagation
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X: NDArray, parameters: Parameters) -> Tuple[NDArray, List[Cache]]:
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(AL: NDArray, Y: NDArray) -> float:
    """
    Implement the cost function.
    
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector, shape (1, number of examples)
    
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    cost = np.squeeze(cost)
    return cost

def sigmoid_backward(dA: NDArray, cache: NDArray) -> NDArray:
    """
    Implement the backward propagation for a single SIGMOID unit.
    
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA: NDArray, cache: NDArray) -> NDArray:
    """
    Implement the backward propagation for a single RELU unit.
    
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_backward(
    dZ: NDArray,
    cache: Tuple[NDArray, NDArray, NDArray]
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Implement the linear portion of backward propagation for a single layer (layer l)
    
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA: NDArray, cache: Cache, activation: str) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL: NDArray, Y: NDArray, caches: List[Cache]) -> Gradients:
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    return grads

def update_parameters(parameters: Parameters, grads: Gradients, learning_rate: float) -> Parameters:
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def predict(X: NDArray, Y: NDArray, parameters: Parameters) -> NDArray:
    """
    This function is used to predict the results of a L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    p = np.zeros((1, m))
    
    # Forward propagation
    probas, _ = L_model_forward(X, parameters)
    
    # Convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    
    # Print accuracy
    print(f"Accuracy: {np.mean(p == Y) * 100}%")
    
    return p

def two_layer_model(X: NDArray, Y: NDArray, layers_dims: Tuple[int, int, int], learning_rate: float = 0.0075, num_iterations: int = 3000, print_cost: bool = False) -> Tuple[Parameters, List[float]]:
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    costs -- list of costs during training
    """
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        
        # Set gradients
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Retrieve updated parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 iterations and for the last iteration
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
            
    return parameters, costs

def L_layer_model(X: NDArray, Y: NDArray, layers_dims: List[int], learning_rate: float = 0.0075, num_iterations: int = 3000, print_cost: bool = False) -> Tuple[Parameters, List[float]]:
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learned by the model
    costs -- costs during training
    """
    np.random.seed(1)
    costs = []
    
    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 iterations and for the last iteration
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
    
    return parameters, costs

def plot_costs(costs: List[float], learning_rate: float = 0.0075) -> None:
    """
    Plot the learning curve with improved styling
    
    Arguments:
    costs -- list of costs during training
    learning_rate -- learning rate used for training
    """
    plt.figure(figsize=(10, 6))
    
    # Plot cost curve
    plt.plot(costs, color='b', label='Training cost')
    
    # Customize plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Iterations (hundreds)', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title(f'Model Learning Curve\nLearning Rate: {learning_rate}', 
              fontsize=14, pad=15)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=10)
    
    # Add textbox with final cost
    final_cost = costs[-1] if costs else 0
    plt.text(0.02, 0.98, f'Final Cost: {final_cost:.6f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def load_data(filename: str = "data.h5") -> Tuple[Optional[NDArray], ...]:
    """
    Load cat vs non-cat dataset
    
    Arguments:
    filename -- path to the dataset file
    
    Returns:
    train_x_orig -- training images
    train_y -- training labels
    test_x_orig -- test images
    test_y -- test labels
    classes -- classes in the dataset
    """
    try:
        train_dataset = h5py.File(filename, "r")
        train_x_orig = np.array(train_dataset["train_set_x"][:])
        train_y = np.array(train_dataset["train_set_y"][:])
        
        test_dataset = h5py.File(filename, "r")
        test_x_orig = np.array(test_dataset["test_set_x"][:])
        test_y = np.array(test_dataset["test_set_y"][:])
        
        classes = np.array(test_dataset["list_classes"][:])
        
        train_y = train_y.reshape((1, train_y.shape[0]))
        test_y = test_y.reshape((1, test_y.shape[0]))
        
        return train_x_orig, train_y, test_x_orig, test_y, classes
    except:
        print("Error: Could not load the dataset file. Make sure the file exists.")
        return None, None, None, None, None

def print_mislabeled_images(
    classes: NDArray,
    X: NDArray,
    Y: NDArray,
    p: NDArray
) -> None:
    """
    Plots images where predictions and truth were different.
    
    Arguments:
    classes -- classes in the dataset
    X -- dataset
    Y -- true labels
    p -- predictions
    """
    a = p + Y
    mislabeled_indices = np.asarray(np.where(a == 1))
    num_images = len(mislabeled_indices[0])
    
    if num_images == 0:
        print("No mislabeled images found")
        return
    
    for i in range(min(num_images, 5)):  # Display up to 5 mislabeled images
        index = mislabeled_indices[1][i]
        
        plt.figure(figsize=(4, 4))
        plt.imshow(X[:,index].reshape(64, 64, 3))
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + 
                  "\nTrue class: " + classes[int(Y[0,index])].decode("utf-8"))
        plt.axis('off')
        plt.show()

def run_demo() -> None:
    """
    Run a demonstration of the deep neural network on the cat vs non-cat dataset
    """
    # Load data
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    
    if train_x_orig is None:
        print("Data loading failed. Exiting demo.")
        return
    
    # Print info about the dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    
    print("Dataset Information:")
    print("-" * 50)
    print(f"Number of training examples: {m_train}")
    print(f"Number of testing examples: {m_test}")
    print(f"Each image is of size: ({num_px}, {num_px}, 3)")
    print(f"train_x_orig shape: {train_x_orig.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"test_x_orig shape: {test_x_orig.shape}")
    print(f"test_y shape: {test_y.shape}")
    print("-" * 50)
    
    # Show an example
    index = 10
    plt.figure(figsize=(4, 4))
    plt.imshow(train_x_orig[index])
    plt.title(f"y = {train_y[0,index]}. It's a {classes[train_y[0,index]].decode('utf-8')} picture.")
    plt.axis('off')
    plt.show()
    
    # Reshape and standardize the data
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    
    print("Pre-processed Data:")
    print("-" * 50)
    print(f"train_x's shape: {train_x.shape}")
    print(f"test_x's shape: {test_x.shape}")
    print("-" * 50)
    
    # Define layer dimensions
    n_x = train_x.shape[0]  # size of input layer
    n_h = 7                 # size of hidden layer
    n_y = 1                 # size of output layer
    
    # 2-layer model training
    print("\nTraining 2-layer model...")
    print("-" * 50)
    
    layers_dims = (n_x, n_h, n_y)
    parameters_2, costs_2 = two_layer_model(train_x, train_y, layers_dims, num_iterations=1500, print_cost=True)
    
    # Plot the cost
    plot_costs(costs_2)
    
    # Predictions
    print("\n2-layer Model Predictions:")
    print("-" * 50)
    print("Training set:")
    pred_train_2 = predict(train_x, train_y, parameters_2)
    print("\nTest set:")
    pred_test_2 = predict(test_x, test_y, parameters_2)
    
    # L-layer model
    print("\nTraining 4-layer model...")
    print("-" * 50)
    
    layers_dims = [n_x, 20, 7, 5, n_y]  # 4-layer model
    parameters_L, costs_L = L_layer_model(train_x, train_y, layers_dims, num_iterations=1500, print_cost=True)
    
    # Plot the cost
    plot_costs(costs_L)
    
    # Predictions
    print("\n4-layer Model Predictions:")
    print("-" * 50)
    print("Training set:")
    pred_train_L = predict(train_x, train_y, parameters_L)
    print("\nTest set:")
    pred_test_L = predict(test_x, test_y, parameters_L)
    
    # Show mislabeled images
    print("\nMislabeled Images from 4-layer Model:")
    print_mislabeled_images(classes, test_x, test_y, pred_test_L)
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo()