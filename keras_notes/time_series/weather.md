# 🌦️ LSTM forecast (single-step)

## Project Overview

Time series forecasting of weather parameters using Long Short-Term Memory (LSTM) neural networks, leveraging the Jena Climate dataset from the Max Planck Institute for Biogeochemistry.

## Dataset Characteristics

### Metadata
- **Source**: Max Planck Institute for Biogeochemistry
- **Location**: Jena, Germany
- **Time Range**: January 10, 2009 - December 31, 2016
- **Observation Frequency**: Every 10 minutes

### Selected Features
1. Pressure (mbar)
2. Temperature (°C)
3. Saturation vapor pressure (mbar)
4. Vapor pressure deficit (mbar)
5. Specific humidity (g/kg)
6. Air density (g/m³)
7. Wind speed (m/s)

## Forecasting Methodology

### Time Series Representation
```math
X_{t} = [x_1, x_2, ..., x_n], \quad t \in \{1, ..., T\}
```
Where:
- $X_t$ represents the time series
- $x_i$ is the value at time step $i$
- $T$ is the total number of time steps

### Data Preprocessing

#### Normalization
```math
X_{normalized} = \frac{X - \mu}{\sigma}
```
- $\mu$: Mean of training data
- $\sigma$: Standard deviation of training data

#### Sequence Preparation
```math
\begin{aligned}
\text{Input Sequence} &= [x_{t-n}, ..., x_{t-1}] \\
\text{Target} &= x_{t+m}
\end{aligned}
```
- Past timestamps: e.g 720 (5 days)
- Future prediction: e.g 72 timestamps (12 hours)
- Sampling rate: 1 observation per hour

## LSTM Neural Network Architecture

### Core Transformation
```math
\begin{aligned}
h_t &= \text{LSTM}(x_t, h_{t-1}) \\
&= f_t \odot \tanh(C_t)
\end{aligned}
```
Where:
- $h_t$: Hidden state
- $f_t$: Forget gate
- $C_t$: Cell state

### Model Structure
```math
\begin{aligned}
\text{Input} &\rightarrow \text{LSTM}(32 \text{ units}) \\
&\rightarrow \text{Dense}(1 \text{ neuron}) \\
&\rightarrow \text{Prediction}
\end{aligned}
```

### Loss Function
```math
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```
Mean Squared Error (MSE) quantifies prediction accuracy:
- $y_i$: Actual temperature
- $\hat{y}_i$: Predicted temperature

## LSTM Internal Mechanics

### Cell Architecture
```math
\begin{aligned}
\text{Input Gate}: i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\text{Forget Gate}: f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
\text{Cell State Update}: \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
\text{Cell State}: C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
\text{Output Gate}: o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
\text{Hidden State}: h_t &= o_t \odot \tanh(C_t)
\end{aligned}
```

### Gate Functions

#### Forget Gate
```math
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
```
- Decides what information to discard from previous cell state
- Sigmoid output between 0 (forget completely) and 1 (keep entirely)

#### Input Gate
```math
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
```
- Determines which new information to store
- Selects relevant features for temperature prediction

#### Cell State Update
```math
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
```
- Creates candidate values to add to cell state
- Uses hyperbolic tangent for non-linear transformation

#### Output Gate
```math
\begin{aligned}
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
```
- Determines what parts of cell state to output
- Generates final hidden state representation

## Data Transformation Stages

### Input Processing
```math
X_{input} \rightarrow \text{Normalization} \rightarrow [x_1, x_2, ..., x_n]
```
- Scales features to [0, 1] range
- Removes feature-specific variations

### Sequence Processing
```math
[x_1, ..., x_{120}] \rightarrow \text{LSTM}(32 \text{ units}) \rightarrow h_{final}
```
- 32 hidden units capture complex temporal dependencies
- Processes entire 120-step sequence
- Extracts most relevant temporal features

### Output Prediction
```math
h_{final} \rightarrow \text{Dense Layer} \rightarrow \hat{y}
```
- Final dense layer produces scalar temperature prediction
- Maps LSTM hidden state to precise temperature value

## Feature Interaction
```math
\text{Temperature} = f(\text{Pressure}, \text{Humidity}, \text{Wind Speed}, ...)
```
- Learns complex, non-linear relationships
- Captures interdependencies between meteorological features

## Usage
```bash
python weather_forecasting.py
```

## Conclusion

The LSTM model transforms raw time series data into meaningful predictions by:
1. Selectively remembering important information
2. Forgetting irrelevant historical patterns
3. Capturing complex temporal dependencies
4. Generating precise, context-aware predictions

## Sources

1. [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
2. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. [Time series weather forecasting in Keras](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)