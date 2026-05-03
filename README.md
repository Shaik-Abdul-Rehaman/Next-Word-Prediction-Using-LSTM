# Next Word Prediction Using LSTM/GRU Networks

A deep learning project that predicts the next word in a sequence using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) neural networks trained on Shakespeare's "Hamlet".

## Overview

This project demonstrates the application of recurrent neural networks (RNNs) for natural language processing tasks. It uses LSTM and GRU architectures to learn the patterns in Shakespeare's text and predict the most likely next word given a sequence of input words.

### Key Features
- **Data Source**: Shakespeare's "Hamlet" from NLTK Gutenberg corpus
- **Models**: Both LSTM and GRU implementations for comparison
- **Early Stopping**: Prevents overfitting during training
- **Text Tokenization**: Converts text to numerical sequences for model processing
- **Sequence Prediction**: Predicts the next word based on input text sequences

## Project Architecture

### 1. **Data Collection**
- Downloads Shakespeare's "Hamlet" from NLTK Gutenberg corpus
- Saves the raw text to `hamlet.txt` for processing

### 2. **Data Preprocessing**
- **Tokenization**: Converts text into numerical tokens using Keras Tokenizer
- **Sequence Creation**: Generates n-gram sequences (1-word, 2-word, ..., up to max length)
- **Padding**: Ensures all sequences have uniform length (14 tokens max)
- **Train-Test Split**: 80% training, 20% testing

**Statistics:**
- Total unique words: 4,818
- Maximum sequence length: 14 tokens
- Total sequences: 25,732

### 3. **Model Architecture**

#### LSTM Model
```
Input Layer (Batch, 13)
↓
Embedding Layer (100 dimensions)
↓
LSTM Layer 1 (100 units, return_sequences=True)
↓
Dropout (0.3)
↓
LSTM Layer 2 (50 units)
↓
Dense Output Layer (4,818 units with Softmax)
```
**Total Parameters**: 838,118 (3.20 MB)

#### GRU Model
```
Input Layer (Batch, 13)
↓
Embedding Layer (100 dimensions)
↓
GRU Layer 1 (100 units, return_sequences=True)
↓
Dropout (0.3)
↓
GRU Layer 2 (50 units)
↓
Dense Output Layer (4,818 units with Softmax)
```
**Total Parameters**: 810,918 (3.09 MB)

### 4. **Model Training**
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50 (with early stopping)
- **Batch Size**: Default (32)
- **Validation Split**: 20%

**Results:**
- Training Accuracy: ~30.09% (at epoch 50)
- Final Training Loss: ~3.41

### 5. **Model Evaluation**
The model can predict the next word from input sequences:

**Example 1:**
- Input: "To be or not to be"
- Predicted Next Word: "captaines"

**Example 2:**
- Input: "Bar. Sit downe a-while, And let vs once againe assaile your eares, That are so fortified against our Story, What we two Nights haue seene Hor. Well, sit we downe, And let vs heare Barnardo"
- Predicted Next Word: "him"

## Quick Start Guide

### Prerequisites
```bash
Python 3.7+
TensorFlow 2.x
Keras
NLTK
NumPy
Pandas
Scikit-learn
```

### Installation

1. **Clone or download the project**
```bash
cd "LSTM RNN"
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install tensorflow keras nltk numpy pandas scikit-learn
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('gutenberg')"
```

### Usage

#### Option 1: Run the Complete Jupyter Notebook
```bash
jupyter notebook experiemnts.ipynb
```
This will execute all cells including data collection, preprocessing, model training, and predictions.

#### Option 2: Use Pre-trained Model
If you have the saved model (`next_word_lstm.h5` or `.keras`), you can load and use it:

```python
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('next_word_lstm.h5')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Prediction function
def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= 14:
        token_list = token_list[-13:]
    token_list = pad_sequences([token_list], maxlen=13, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Use it
next_word = predict_next_word("to be or")
print(f"Predicted next word: {next_word}")
```

## Project Files

- **`experiemnts.ipynb`**: Main Jupyter notebook containing all code
- **`next_word_lstm.h5`**: Trained LSTM model (if available)
- **`next_word_lstm_model_with_early_stopping.h5`**: LSTM model with early stopping
- **`tokenizer.pickle`**: Saved tokenizer for text-to-token conversion
- **`hamlet.txt`**: Raw Shakespeare's Hamlet text
- **`README.md`**: This file

## Key Concepts Explained

### LSTM vs GRU
- **LSTM** (Long Short-Term Memory): More complex, with separate input/output gates and cell state
- **GRU** (Gated Recurrent Unit): Simplified version of LSTM with fewer parameters

### Embedding Layer
Converts integer token indices to dense vectors of fixed size (100 dimensions), capturing semantic relationships between words.

### Dropout
Randomly deactivates 30% of neurons during training to prevent overfitting.

### Early Stopping
Monitors validation loss and stops training if it doesn't improve for 3 consecutive epochs.

## Performance Analysis

### Model Comparison
- **LSTM Parameters**: 838,118
- **GRU Parameters**: 810,918
- **Difference**: GRU is ~3% lighter with fewer parameters
- **Training Time**: GRU typically trains faster due to simpler architecture

### Limitations
- Limited vocabulary (4,818 unique words)
- Relatively short sequence context (max 14 words)
- Shakespeare's archaic language makes modern text prediction less accurate
- Model accuracy ~30% due to complexity of language patterns

### Improvements for Production
1. Use larger datasets (news articles, books)
2. Implement beam search for better predictions
3. Use pre-trained embeddings (Word2Vec, GloVe)
4. Ensemble multiple models
5. Deploy with Streamlit or Flask web interface

## Troubleshooting

### Issue: Model not found error
**Solution**: Ensure you've trained the model or have the pre-trained weights in the correct directory.

### Issue: NLTK data missing
**Solution**: Run `python -c "import nltk; nltk.download('gutenberg')"`

### Issue: Out of memory during training
**Solution**: Reduce batch size or use smaller sequence length.

### Issue: Low prediction accuracy
**Solution**: This is expected with Shakespeare's archaic text. Modern literature would yield better results.

## Future Enhancements

1. **Web Interface**: Build a Streamlit or Flask app for interactive predictions
2. **Multi-language Support**: Extend to other languages and texts
3. **Attention Mechanism**: Implement attention for better context understanding
4. **Transformer Models**: Experiment with BERT or GPT-based approaches
5. **Hyperparameter Tuning**: Optimize embedding dimensions, LSTM units, dropout rates
6. **Real-time Prediction**: Deploy as an API service

## References

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [LSTM Networks - Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [GRU Architecture](https://arxiv.org/abs/1409.0473)
- [NLTK Book - NLP Fundamentals](https://www.nltk.org/book/)

## Author
Rehaman

## License
Open source - feel free to use for learning and experimentation.

---

**Last Updated**: 2026-05-03
