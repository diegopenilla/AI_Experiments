import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
from typing import Tuple, Callable

def load_mnist_data(num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess MNIST dataset.
    
    Args:
        num_classes: Number of digit classes to classify
    
    Returns:
        Tuple of preprocessed training and test data and labels
    """
    # Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # Reshape to add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test

def create_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """
    Create a Convolutional Neural Network for MNIST classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    return model

def train_model(
    model: keras.Model, 
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    batch_size: int = 64, 
    epochs: int = 10
) -> keras.callbacks.History:
    """
    Train the Keras model on MNIST dataset.
    
    Args:
        model: Keras model to train
        x_train: Training image data
        y_train: Training labels
        batch_size: Number of samples per gradient update
        epochs: Number of training epochs
    
    Returns:
        Training history
    """
    model.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )
    
    return model.fit(
        x_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_split=0.1
    )

def evaluate_model(
    model: keras.Model, 
    x_test: np.ndarray, 
    y_test: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model
        x_test: Test image data
        y_test: Test labels
    
    Returns:
        Tuple of test loss and test accuracy
    """
    return model.evaluate(x_test, y_test, verbose=0)

def main() -> None:
    """
    Main function to run MNIST CNN classification pipeline.
    """
    # Model configuration
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_mnist_data(num_classes)
    
    # Create and train model
    model = create_cnn_model(input_shape, num_classes)
    model.summary()
    
    # Train the model
    history = train_model(model, x_train, y_train)
    
    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()