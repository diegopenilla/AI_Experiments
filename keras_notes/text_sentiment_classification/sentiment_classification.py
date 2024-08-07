import os
import re
import string
from typing import Tuple, Any, Callable, Dict, Optional
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.utils import text_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Global variables for model components
vectorize_layer = None
trained_model = None
end_to_end_model = None


def download_and_extract_dataset(data_dir: str = "aclImdb") -> None:
    """
    Download and extract the IMDB dataset if it doesn't exist.
    
    Args:
        data_dir: Directory where the dataset should be located
    """
    # Check if data directory already exists
    if not Path(data_dir).exists():
        print("Downloading IMDB dataset...")
        # Download the dataset
        os.system("curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
        
        # Extract the dataset
        print("Extracting dataset...")
        os.system("tar -xf aclImdb_v1.tar.gz")
        
        # Clean up the tar file
        os.system("rm aclImdb_v1.tar.gz")
        print("Dataset ready.")
    else:
        print(f"Dataset directory {data_dir} already exists. Skipping download.")


def custom_standardization(input_data: tf.Tensor) -> tf.Tensor:
    """
    Standardize text by lowercasing, removing HTML break tags and punctuation.
    
    Args:
        input_data: Raw text input
        
    Returns:
        Standardized text
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


def vectorize_text(text: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convert text to integer sequences using the vectorization layer.
    
    Args:
        text: Text input
        label: Label input
        
    Returns:
        Tuple of vectorized text and label
    """
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def create_datasets(
    data_dir: str,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 1337
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create train, validation, and test datasets.
    
    Args:
        data_dir: Directory containing the IMDB dataset
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    # Clean up unsupervised data if it exists
    unsup_dir = os.path.join(data_dir, "train/unsup")
    if os.path.exists(unsup_dir):
        import shutil
        shutil.rmtree(unsup_dir)
    
    # Create raw datasets
    raw_train_ds = text_dataset_from_directory(
        os.path.join(data_dir, "train"),
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )
    
    raw_val_ds = text_dataset_from_directory(
        os.path.join(data_dir, "train"),
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )
    
    raw_test_ds = text_dataset_from_directory(
        os.path.join(data_dir, "test"),
        batch_size=batch_size
    )
    
    return raw_train_ds, raw_val_ds, raw_test_ds


def create_text_vectorization_layer(
    train_ds: tf.data.Dataset,
    max_features: int,
    sequence_length: int
) -> layers.Layer:
    """
    Create and adapt a TextVectorization layer.
    
    Args:
        train_ds: Training dataset
        max_features: Maximum number of tokens in vocabulary
        sequence_length: Length of output sequences
        
    Returns:
        Adapted TextVectorization layer
    """
    global vectorize_layer
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    
    # Create text-only dataset and adapt the layer
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)
    
    return vectorize_layer


def build_model(
    max_features: int,
    embedding_dim: int
) -> keras.Model:
    """
    Build a 1D CNN model for sentiment classification.
    
    Args:
        max_features: Size of vocabulary
        embedding_dim: Dimensionality of embedding
        
    Returns:
        Compiled Keras model
    """
    # Integer input for vocab indices
    inputs = keras.Input(shape=(None,), dtype="int64")
    
    # Increased embedding dimension for better representation
    x = layers.Embedding(max_features + 1, embedding_dim)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)  # Better dropout for sequences
    
    # Modified CNN architecture
    x = layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2)(x)
    x = layers.Conv1D(128, 5, padding="valid", activation="relu", strides=2)(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # Added batch normalization
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    
    # Create and compile model
    model = keras.Model(inputs, predictions)
    
    # Use Adam optimizer with custom learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    return model


def create_end_to_end_model(
    base_model: keras.Model,
    vectorize_layer: layers.Layer
) -> keras.Model:
    """
    Create an end-to-end model that processes raw strings.
    
    Args:
        base_model: Trained model that processes vectorized inputs
        vectorize_layer: TextVectorization layer
        
    Returns:
        End-to-end model
    """
    # String input
    inputs = keras.Input(shape=(1,), dtype="string")
    
    # Turn strings into vocab indices
    indices = vectorize_layer(inputs)
    
    # Turn vocab indices into predictions
    outputs = base_model(indices)
    
    # Create and compile end-to-end model
    end_to_end_model = keras.Model(inputs, outputs)
    end_to_end_model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    return end_to_end_model


def save_models(model_dir: str = "saved_models") -> None:
    """
    Save the trained models for later use.
    
    Args:
        model_dir: Directory to save models
    """
    global trained_model, end_to_end_model
    
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save models
    if trained_model is not None:
        trained_model.save(os.path.join(model_dir, "base_model"))
    
    if end_to_end_model is not None:
        end_to_end_model.save(os.path.join(model_dir, "end_to_end_model"))
    
    print(f"Models saved to {model_dir}")


def load_models(model_dir: str = "saved_models") -> Tuple[Optional[keras.Model], Optional[keras.Model]]:
    """
    Load previously saved models.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        Tuple of (base_model, end_to_end_model)
    """
    base_model_path = os.path.join(model_dir, "base_model")
    end_to_end_model_path = os.path.join(model_dir, "end_to_end_model")
    
    base_model = None
    e2e_model = None
    
    if os.path.exists(base_model_path):
        base_model = keras.models.load_model(base_model_path)
        print("Base model loaded successfully")
    
    if os.path.exists(end_to_end_model_path):
        e2e_model = keras.models.load_model(end_to_end_model_path)
        print("End-to-end model loaded successfully")
    
    return base_model, e2e_model


def predict_sentiment(model: keras.Model, text: str) -> Tuple[float, str]:
    """
    Predict sentiment for a given text.
    
    Args:
        model: Trained end-to-end model
        text: Text to analyze
        
    Returns:
        Tuple of (probability, sentiment label)
    """
    # Convert text to tensor
    text_tensor = tf.constant([text])
    
    # Make prediction
    probability = float(model.predict(text_tensor, verbose=0)[0][0])
    sentiment = "positive" if probability >= 0.5 else "negative"
    return probability, sentiment

def test_model_predictions(model: keras.Model) -> None:
    """
    Test the model with example reviews and print results.
    """
    print("\nTesting Model Predictions:")
    print("-" * 10)
    
    # Test reviews with different sentiments
    example_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Terrible waste of time. Poor acting and boring plot.",
        "An okay film, nothing special but also not terrible.",
        "The best movie I've seen this year! Incredible performances!",
        "I fell asleep halfway through. Very disappointing.",
    ]
    
    for review in example_reviews:
        try:
            probability, sentiment = predict_sentiment(model, review)
            
            # Print formatted results
            print(f"\nReview: {review}")
            print(f"Sentiment: {sentiment.upper()}")
            print(f"Confidence: {probability:.2%}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing review: {e}")
            continue

def interactive_testing(model: keras.Model) -> None:
    """
    Allow user to enter reviews and get sentiment predictions.
    
    Args:
        model: Trained end-to-end model
    """
    print("\nInteractive Testing:")
    print("Enter reviews (or 'quit' to exit):")
    
    while True:
        try:
            user_review = input("\nEnter a review: ").strip()
            if user_review.lower() in ['quit', 'exit', 'q']:
                break
                
            probability, sentiment = predict_sentiment(model, user_review)
            print(f"Sentiment: {sentiment.upper()}")
            print(f"Confidence: {probability:.2%}")
        except Exception as e:
            print(f"Error processing review: {e}")
            continue


def main(data_dir: str = "aclImdb", train_new: bool = True) -> None:
    """
    Main function to run the IMDB sentiment analysis pipeline.
    
    Args:
        data_dir: Directory containing the IMDB dataset
        train_new: Whether to train a new model or load existing models
    """
    global trained_model, end_to_end_model
    
    batch_size = 64  
    max_features = 15000  
    embedding_dim = 200 
    sequence_length = 200  
    epochs = 10 
    
    if train_new:
        # Download and extract dataset if needed
        download_and_extract_dataset(data_dir)
        
        # Create datasets
        raw_train_ds, raw_val_ds, raw_test_ds = create_datasets(
            data_dir=data_dir,
            batch_size=batch_size
        )
        
        # Log dataset information
        print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
        print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
        print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")
        
        # Create and adapt vectorization layer
        create_text_vectorization_layer(
            raw_train_ds,
            max_features,
            sequence_length
        )
        
        # Vectorize the datasets
        train_ds = raw_train_ds.map(vectorize_text)
        val_ds = raw_val_ds.map(vectorize_text)
        test_ds = raw_test_ds.map(vectorize_text)
        
        # Optimize datasets for GPU training
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Build and train model
        trained_model = build_model(max_features, embedding_dim)
        
        # Add callbacks for better training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                mode='max'
            ),
            # Reduce learning rate when training plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001
            ),
            # Save best model
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train with callbacks
        history = trained_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = trained_model.evaluate(test_ds)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Create end-to-end model for inference
        end_to_end_model = create_end_to_end_model(trained_model, vectorize_layer)
        
        # Evaluate end-to-end model
        end_to_end_loss, end_to_end_accuracy = end_to_end_model.evaluate(raw_test_ds)
        print(f"End-to-end test accuracy: {end_to_end_accuracy:.4f}")
        
        # Save models for later use
        # save_models()
    else:
        # Load previously trained models
        trained_model, end_to_end_model = load_models()
        
        if end_to_end_model is None:
            print("No saved models found. Please train a new model first.")
            return
    
    # Test the model with example reviews
    test_model_predictions(end_to_end_model)
    
    # Interactive testing
    interactive_testing(end_to_end_model)


if __name__ == "__main__":
    # Run the main training pipeline with a new model
    # To use a saved model instead, set train_new=False
    main(train_new=True)