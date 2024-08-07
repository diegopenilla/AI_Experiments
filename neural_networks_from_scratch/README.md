# 🧠 Neural Networks Foundations - Classifier

Neural networks process input data through interconnected layers, transforming inputs via weighted sums and activation functions. Forward propagation computes predictions by passing signals through hidden layers (using ReLU) and output layers (using sigmoid).  

The binary cross-entropy cost quantifies prediction errors by comparing outputs to true labels. Backpropagation calculates gradients to adjust weights and biases, minimizing errors through gradient descent. Finally, predictions are made by thresholding output probabilities, and accuracy measures classification performance.

### **Linear Computation**  
The linear transformation for layer \( l \):  
```math
Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}
```
Each neuron computes a weighted sum of inputs from the previous layer plus a bias term.

### **Activation Functions**  
- **Sigmoid**:  
  ```math
  \sigma(Z) = \frac{1}{1 + e^{-Z}}
  ```  
  Sigmoid squeezes any input into a value between 0 and 1, making it perfect for binary classification outputs.

- **ReLU**:  
  ```math
  \text{ReLU}(Z) = \max(0, Z)
  ```  
  ReLU passes positive values unchanged while setting negative values to zero, which helps prevent vanishing gradients.

### **Layer Activations**  
- **Hidden layer**:  
  ```math
  A^{[l]} = \text{ReLU}(Z^{[l]})
  ```  
  Hidden layers typically use ReLU to introduce non-linearity while maintaining efficient gradient flow.

- **Output layer**:  
  ```math
  A^{[L]} = \sigma(Z^{[L]})
  ```  
  The output layer uses sigmoid to produce probability values for binary classification tasks.

---

## Cost Function

### **Binary Cross-Entropy**  
```math
J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log(a^{[L](i)}) + (1-y^{(i)})\log(1-a^{[L](i)}) \right]
```
This function quantifies prediction errors by heavily penalizing confident but wrong predictions.

---

## Backward Propagation

### **Output Layer Gradient**  
- Gradient of activation:  
  ```math
  dA^{[L]} = -\left( \frac{Y}{A^{[L]}} - \frac{1-Y}{1-A^{[L]}} \right)
  ```  
  This formula calculates how much the output probabilities need to change to better match true labels.

- Gradient of linear transformation:  
  ```math
  dZ^{[L]} = dA^{[L]} \cdot A^{[L]} \cdot (1 - A^{[L]})
  ```  
  The pre-activation gradient incorporates the sigmoid derivative to determine how Z values should change.

### **Hidden Layer Gradients**  
- Gradient of activation:  
  ```math
  dA^{[l]} = W^{[l+1]T} \cdot dZ^{[l+1]}
  ```  
  This propagates error information backward from later layers to earlier ones in the network.

- Gradient of linear transformation:  
  ```math
  dZ^{[l]} = dA^{[l]} \cdot \mathbb{I}(Z^{[l]} > 0)
  ```  
  For ReLU layers, gradients only flow through neurons that were active during forward propagation.

### **Parameter Gradients**  
- Weights:  
  ```math
  dW^{[l]} = \frac{1}{m} dZ^{[l]} \cdot A^{[l-1]T}
  ```  
  This computes how each weight should change by considering both the error signal and the corresponding input.

- Biases:  
  ```math
  db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l](i)}
  ```  
  Bias gradients are calculated by averaging the pre-activation gradients across all training examples.

---

## Parameter Updates

### **Gradient Descent**  
- Weights update:  
  ```math
  W^{[l]} := W^{[l]} - \alpha \cdot dW^{[l]}
  ```  
  Weights are adjusted in the direction that reduces the cost, with the learning rate controlling step size.

- Biases update:  
  ```math
  b^{[l]} := b^{[l]} - \alpha \cdot db^{[l]}
  ```  
  Biases are updated similarly to weights, moving in the direction that minimizes prediction errors.

---

## Prediction

### **Binary Classification**  
```math
\hat{y}^{(i)} = \begin{cases} 
1 & \text{if } a^{[L](i)} > 0.5 \\
0 & \text{otherwise}
\end{cases}
```  
The model predicts the positive class when output probability exceeds 0.5, and the negative class otherwise.

### **Accuracy**  
```math
\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}(y^{(i)} = \hat{y}^{(i)}) \times 100\%
```
Accuracy measures the percentage of correct predictions across all examples in the dataset.


## Regularization Techniques

### **L2 Regularization**

The standard way to avoid overfitting is called **L2 regularization**. It consists of appropriately modifying your cost function, from:
```math

J = -\frac{1}{m} \sum_{i = 1}^{m} \large{(} y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} \tag{1}
```
To:
```math 
J_{regularized} = -\frac{1}{m} \sum_{i = 1}^{m} \large{(} y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} + \frac{1}{m} \frac{\lambda}{2} \sum_l\sum_k\sum_j W_{k,j}^{[l]2} \tag{2}
```

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. By penalizing the square values of the weights in the cost function you drive all the weights to smaller values. 
- It becomes too costly for the cost to have large weights! 
- Weights end up smaller, leading to a smoother model in which the output changes more slowly as the input changes.


## Gradient Checking

Backpropagation computes the gradients $\frac{\partial J}{\partial \theta}$, where $\theta$ denotes the parameters of the model. $J$ is computed using forward propagation and your loss function.

Because forward propagation is relatively easy to implement, you're confident you got that right, and so you're almost 100% sure that you're computing the cost $J$ correctly. Thus, you can use your code for computing $J$ to verify the code for computing $\frac{\partial J}{\partial \theta}$.

Let's look back at the definition of a derivative (or gradient):

```math
\frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} 
```

