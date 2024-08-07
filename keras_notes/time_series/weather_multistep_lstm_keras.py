from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from zipfile import ZipFile

class WeatherTimeSeriesForecaster:
    def __init__(
        self, 
        data_url: str = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
        split_fraction: float = 0.715,
        past_timestamps: int = 720,
        future_timestamps: int = 72,
        sampling_step: int = 6,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        epochs: int = 3
    ):
        """
        Initialize weather time series forecasting model for multi-step predictions.

        Args:
            data_url: URL to download climate dataset
            split_fraction: Fraction of data to use for training
            past_timestamps: Number of past timestamps to use for prediction
            future_timestamps: Number of future timestamps to predict
            sampling_step: Sampling rate (observations per hour)
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            epochs: Number of training epochs
        """
        self.data_url = data_url
        self.split_fraction = split_fraction
        self.past_timestamps = past_timestamps
        self.future_timestamps = future_timestamps
        self.sampling_step = sampling_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # We will predict the temperature column (index 1) for multiple future steps.
        self.feature_keys = [
            "p (mbar)", "T (degC)", "VPmax (mbar)", 
            "VPdef (mbar)", "sh (g/kg)", "rho (g/m**3)", "wv (m/s)"
        ]

        self.model: Optional[keras.Model] = None
        self.dataset_train: Optional[tf.data.Dataset] = None
        self.dataset_val: Optional[tf.data.Dataset] = None

    def download_and_load_data(self) -> pd.DataFrame:
        """
        Download and load Jena Climate dataset.

        Returns:
            Pandas DataFrame with climate data
        """
        zip_path = keras.utils.get_file(
            origin=self.data_url,
            fname="jena_climate_2009_2016.csv.zip"
        )
        with ZipFile(zip_path) as zip_file:
            zip_file.extractall()
        return pd.read_csv("jena_climate_2009_2016.csv")

    def normalize_data(self, data: np.ndarray, train_split: int) -> np.ndarray:
        """
        Normalize data by subtracting mean and dividing by standard deviation.

        Args:
            data: Input data array
            train_split: Index to split training data

        Returns:
            Normalized data array
        """
        data_mean = data[:train_split].mean(axis=0)
        data_std = data[:train_split].std(axis=0)
        return (data - data_mean) / data_std

    def create_multi_step_sequences(
        self, data: np.ndarray, start_index: int, end_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create multi-step input (X) and output (Y) sequences.

        Each sample X consists of `past_timestamps / sampling_step` rows.
        Each target Y is a vector of length `future_timestamps` for the temperature column.

        Args:
            data: Normalized data array of shape [num_samples, num_features]
            start_index: Where to start indexing in data
            end_index: Where to stop indexing in data

        Returns:
            A tuple of (X, Y) NumPy arrays.
        """
        X, Y = [], []
        sequence_length = self.past_timestamps // self.sampling_step
        # We'll predict only the temperature column => index 1
        for i in range(start_index, end_index - self.future_timestamps):
            # i is the last index of the input window
            # The window for X is [i - self.past_timestamps, i), stepped by sampling_step
            start_x = i - self.past_timestamps
            end_x = i
            indices_x = range(start_x, end_x, self.sampling_step)

            # Prepare input
            seq_x = data[indices_x]

            # Prepare multi-step target
            seq_y = data[i : i + self.future_timestamps, 1]  # temperature column

            X.append(seq_x)
            Y.append(seq_y)

        return np.array(X), np.array(Y)

    def prepare_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare training and validation datasets for multi-step predictions.

        Returns:
            Tuple of training and validation datasets
        """
        df = self.download_and_load_data()
        features = df[self.feature_keys]

        train_split = int(self.split_fraction * len(df))
        data = features.values

        # Normalize
        data = self.normalize_data(data, train_split)

        # Create train & validation splits
        train_data = data[:train_split]
        val_data = data[train_split:]

        # We'll create X, Y sequences for both train and val sets.
        x_train, y_train = self.create_multi_step_sequences(
            train_data, 
            start_index=self.past_timestamps, 
            end_index=train_data.shape[0]
        )
        
        x_val, y_val = self.create_multi_step_sequences(
            val_data, 
            start_index=self.past_timestamps, 
            end_index=val_data.shape[0]
        )

        # Convert to tf.data.Dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset_train = dataset_train.batch(self.batch_size).shuffle(1000)

        dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        dataset_val = dataset_val.batch(self.batch_size)

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        return dataset_train, dataset_val

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM-based time series forecasting model for multi-step outputs.

        Args:
            input_shape: Shape of input data (sequence_length, num_features)

        Returns:
            Compiled Keras model
        """
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.LSTM(32, return_sequences=True)(inputs)
        x = keras.layers.LSTM(16)(x)
        # Output dimension is future_timestamps
        outputs = keras.layers.Dense(self.future_timestamps)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse"
        )
        self.model = model
        return model

    def train_model(self) -> keras.callbacks.History:
        """
        Train the multi-step time series forecasting model.

        Returns:
            Training history
        """
        if self.dataset_train is None or self.dataset_val is None:
            self.prepare_datasets()

        # Get input shape from one batch
        for batch in self.dataset_train.take(1):
            inputs, _ = batch
            input_shape = inputs.shape[1:]  # (sequence_length, num_features)

        model = self.build_model(input_shape)

        es_callback = keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            self.dataset_train,
            epochs=self.epochs,
            validation_data=self.dataset_val,
            callbacks=[es_callback]
        )
        return history

    def visualize_loss(self, history: keras.callbacks.History) -> None:
        """
        Visualize training and validation loss.

        Args:
            history: Training history object
        """
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, label="Training loss")
        plt.plot(epochs, val_loss, label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def predict_and_visualize(self, num_predictions: int = 1) -> None:
        """
        Make predictions and visualize multi-step results.

        Args:
            num_predictions: Number of samples to visualize from the validation set
        """
        if self.model is None:
            raise ValueError("Model not found. Train or load the model first.")
        if self.dataset_val is None:
            raise ValueError("Validation dataset not found. Prepare datasets first.")

        # Collect a few samples from the validation set
        val_samples = []
        for batch_in, batch_out in self.dataset_val.take(num_predictions):
            val_samples.append((batch_in.numpy(), batch_out.numpy()))

        for i, (x, y_true) in enumerate(val_samples):
            # We'll just use the first sample in the batch to visualize
            x_sample = x[0:1]
            y_sample_true = y_true[0]
            y_sample_pred = self.model.predict(x_sample)[0]

            plt.figure(figsize=(12, 6))
            plt.plot(
                range(self.future_timestamps), 
                y_sample_true, 
                "ro-",
                label="True Future"
            )
            plt.plot(
                range(self.future_timestamps), 
                y_sample_pred, 
                "go-",
                label="Predicted Future"
            )
            plt.title(f"Multi-Step Prediction Sample {i+1}")
            plt.xlabel("Future Time Steps")
            plt.ylabel("Normalized Temperature")
            plt.legend()
            plt.show()

def main():
    """
    Main execution function for weather time series forecasting (multi-step).
    """
    # Example usage
    forecaster = WeatherTimeSeriesForecaster(
        epochs=10,
        batch_size=128,   # Reduced batch size
        learning_rate=0.001,
        past_timestamps=50,   # e.g. 720 minutes (12 hours) of history
        future_timestamps=3,  # e.g. 72 minutes (1.2 hours) of future
        sampling_step=6        # sample once every hour
    )

    try:
        # Prepare datasets
        train_ds, val_ds = forecaster.prepare_datasets()

        # Train model
        history = forecaster.train_model()

        # Visualize training loss
        forecaster.visualize_loss(history)

        # Make multi-step predictions (visualize for 2 samples)
        forecaster.predict_and_visualize(num_predictions=2)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()