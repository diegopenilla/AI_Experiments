# 🧠 Sentiment Analysis - Deep Learning Approach

Sentiment analysis using a 1D Convolutional Neural Network (1D CNN) for binary classification of movie reviews from the IMDB dataset.

## Network Architecture

### **Text Preprocessing**
Text preprocessing transforms raw text into a format suitable for neural network processing:

```math
\text{Preprocessing}(X) = \left\{
\begin{array}{l}
\text{Lowercase Conversion} \\
\text{HTML Tag Removal} \\
\text{Punctuation Stripping} \\
\text{Tokenization} \\
\text{Vectorization}
\end{array} \right.
```

### **Text Vectorization Layer**
Converts text into integer sequences:
```math
\text{Vectorize}(X) = [t_1, t_2, ..., t_n]
```
Where:
- $t_i$ represents token indices
- Maximum vocabulary size: 15,000 tokens
- Fixed sequence length: 200 tokens

### **Embedding Layer**
Transforms token indices into dense vector representations:
```math
\text{Embedding}(X) = [v_1, v_2, ..., v_n]
```
Key parameters:
- Embedding dimension: 200
- Learns semantic representations of tokens

### **Convolutional Layers**
Extract local patterns from text sequences:
```math
\text{Conv1D}(X) = \text{ReLU}(W * X + b)
```
Architecture:
- First layer: 64 filters, kernel size 5
- Second layer: 128 filters, kernel size 5
- Stride of 2 for dimensional reduction
- ReLU activation for non-linearity

### **Pooling and Regularization**
```math
\begin{aligned}
\text{SpatialDropout1D}(X, p) &= X \odot \text{Bernoulli}(1-p) \\
\text{GlobalMaxPooling1D}(X) &= \max(X_1, X_2, ..., X_n)
\end{aligned}
```
- Dropout rate: 0.2
- Global max pooling reduces sequence to fixed-size representation

### **Dense Output Layer**
Produces binary sentiment probability:
```math
\text{Dense}(X) = \text{Sigmoid}(WX + b)
```
- Single neuron with sigmoid activation
- Output range: [0, 1]
  - > 0.5: Positive sentiment
  - ≤ 0.5: Negative sentiment

## Training Process

### **Loss Function: Binary Cross-Entropy**
```math
L = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
```
Where:
- $y_i$: True label (0 or 1)
- $\hat{y}_i$: Predicted probability

### **Optimization: Adam**
Adaptive learning rate optimization:
- Initial learning rate: 0.001
- Adaptive moment estimation
- Bias correction for faster convergence

### **Training Configuration**
- **Batch Size**: 64 samples
- **Epochs**: 10
- **Validation Split**: 20%
- **Early Stopping**: Monitors validation accuracy
- **Learning Rate Reduction**: Adapts during training plateaus

## Model Evaluation Metrics

### **Accuracy**
```math
\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}(\text{sign}(\hat{y}_i) = y_i)
```

### **Prediction Confidence**
```math
\text{Confidence} = |\hat{y}_i - 0.5| \times 2
```
- Higher confidence indicates stronger sentiment polarity

### **End-to-End Model**
Combines text vectorization and sentiment prediction:
1. Raw text input
2. Automatic tokenization
3. Vectorization
4. Sentiment prediction

## Dataset: IMDB Movie Reviews

The IMDB dataset is a widely-used benchmark collection for sentiment analysis:

- **Size**: 50,000 movie reviews
  - 25,000 for training
  - 25,000 for testing
- **Balance**: Evenly split between positive and negative reviews
- **Labels**:
  - Positive: Reviews with score ≥ 7/10
  - Negative: Reviews with score ≤ 4/10
  - Neutral reviews (5-6/10) are not included
- **Characteristics**:
  - Reviews are pre-processed to maintain only ASCII characters
  - Maximum of 30 reviews per movie to prevent bias
  - Reviews vary in length from short sentences to multiple paragraphs
  - Dataset includes various movie genres and time periods
- **[Source](https://huggingface.co/datasets/stanfordnlp/imdb)**: Originally collected from IMDB.com for academic purposes by Stanford researchers


## Execution

```bash
python3 sentiment_analysis.py
```

The 1D Convolutional Neural Network progressively learns semantic meaning through specialized layers:

**1. Embedding Layer: Semantic Representation**
- Transforms words into dense vector representations
- Captures semantic relationships between words
- Example: "good" and "excellent" get similar vector representations

**2. Convolutional Layers: Local Pattern Detection**
- First layer identifies basic linguistic patterns
  - Detects simple word combinations
  - Recognizes common sentiment indicators
- Second layer combines these patterns
  - Identifies more complex sentiment signals

**3. Global Max Pooling: Essential Meaning Extraction**
- Distills most significant semantic features
- Captures the "essence" of the review
- Removes positional information

**4. Dense Layers: Sentiment Reasoning**
- Learns abstract representations of sentiment
- Combines extracted features into final prediction
- Determines overall sentiment probability

### Meaning Extraction Mechanism

```
Raw Text → Word Embeddings → Local Patterns → 
Complex Patterns → Essential Features → 
Sentiment Probability
```

**Example Learning Process:**
- Input: "This movie was incredibly awesome!"
- Embedding: Converts words to semantic vectors
- Convolutions: Detect positive intensity markers
- Pooling: Extracts most significant sentiment signals
- Output: High positive sentiment probability (> 0.9)



### Inference
```python
# Predict sentiment for a review
text = "This movie was fantastic!"
probability, sentiment = predict_sentiment(end_to_end_model, text)
```

### Sources

1. [Sentiment Classification in Keras](https://keras.io/examples/nlp/text_classification_from_scratch/)
2. [IMDB Movie Reviews Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
3. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)