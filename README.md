# Sentiment Analysis with LSTM and Pre-trained Word Embeddings (GloVe) in TensorFlow

This project demonstrates how to perform sentiment analysis on the IMDB movie reviews dataset using a Long Short-Term Memory (LSTM) network, with a particular focus on leveraging pre-trained word embeddings (GloVe) to potentially enhance model performance. It also includes a comparison with an LSTM model that learns embeddings from scratch.

## Project Overview

The Python script performs the following key steps:

1.  **Dataset Loading and Preprocessing**: Loads the IMDB dataset, which is pre-tokenized. It also includes code to decode sample reviews back into human-readable text.
2.  **Sequence Padding**: Pads all movie review sequences to a uniform length.
3.  **GloVe Embeddings Loading**: Loads pre-trained GloVe word vectors from a text file.
4.  **Embedding Matrix Preparation**: Creates an embedding matrix that maps words in the IMDB vocabulary to their corresponding GloVe vectors.
5.  **LSTM Model with Pre-trained Embeddings**: Defines an LSTM model where the `Embedding` layer is initialized with the pre-trained GloVe matrix and set to be non-trainable.
6.  **LSTM Model without Pre-trained Embeddings**: Defines a standard LSTM model where the `Embedding` layer learns embeddings from scratch.
7.  **Model Compilation and Training**: Compiles and trains both LSTM models on the IMDB dataset.
8.  **Model Evaluation and Comparison**: Evaluates both models on the test set and compares their final accuracies.
9.  **Visualization**: Plots a bar chart to visually compare the test accuracies of the two models.

## Dataset

The **IMDB movie reviews dataset** is a widely used benchmark for binary sentiment classification. It contains 50,000 highly polarized movie reviews (25,000 for training, 25,000 for testing), labeled as either positive (1) or negative (0). The dataset is pre-processed, with reviews already converted into sequences of integers, where each integer represents a specific word.

### Data Preprocessing

* **Vocabulary Size (`vocab_size`)**: Set to 10,000, considering only the most frequent words.
* **Maximum Sequence Length (`max_len`)**: Set to 200. All movie review sequences are padded with zeros (`padding="post"`) or truncated to this fixed length.
* **Review Decoding**: The script includes a snippet to decode the first few integer-encoded reviews back to plain text, illustrating the mapping of integers to words.

## Word Embeddings: Learning vs. Pre-trained (GloVe)

Word embeddings are dense vector representations of words. They capture semantic relationships between words, meaning words with similar meanings tend to have similar vector representations.

### 1. Learning Embeddings from Scratch (LSTM without GloVe)

In the "LSTM without GloVe" model, the `tf.keras.layers.Embedding` layer is initialized randomly. During the training process, the neural network learns these word embeddings as part of optimizing the overall model for the sentiment analysis task. This approach works well when you have a large dataset.

### 2. Using Pre-trained Embeddings (LSTM with GloVe)

Pre-trained word embeddings are word vectors that have been learned on very large text corpora (e.g., billions of words from Wikipedia, common crawl). They have already captured general semantic and syntactic relationships between words.

This project utilizes **GloVe (Global Vectors for Word Representation)** embeddings. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

**How it's used in this code:**

* **Downloading GloVe**: You need to download the GloVe embeddings file. For this project, the `glove.6B.100d.txt` file is used, which contains 100-dimensional embeddings trained on a 6 Billion token corpus.
    * **Download Link**: You can download `glove.6B.zip` directly from Stanford NLP's website: **[http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)**
    * **Unzip the file** after downloading it. Make sure `glove.6B.100d.txt` is accessible in your project directory (or specify its full path).
* **Loading GloVe vectors**: The script reads `glove.6B.100d.txt` and parses each line to create a dictionary (`embedding_index`) where keys are words and values are their corresponding 100-dimensional GloVe vectors.
* **Preparing Embedding Matrix**: An `embedding_matrix` is created. For each word in the IMDB vocabulary (up to `vocab_size`), if a corresponding GloVe vector exists, it is placed into this matrix at the word's index. Words not found in GloVe (or rare words outside `vocab_size`) will have a row of zeros.
* **`Embedding` Layer Initialization**: In the "LSTM with GloVe" model, the `Embedding` layer is initialized with this `embedding_matrix`:
    `Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)`
    * `weights=[embedding_matrix]`: This tells Keras to initialize the embedding layer's weights with our pre-trained GloVe vectors.
    * `trainable=False`: This is crucial. It means the pre-trained embeddings will not be updated during the model's training. They are used as fixed, powerful features. Setting `trainable=True` would allow fine-tuning the embeddings during training, which can be beneficial if your dataset is large enough.

## Model Architectures

Both models use an LSTM layer for sequential processing and a `Dense` layer for binary classification.

### LSTM Model (Learned Embeddings)

```python
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128), # Embeddings learned from scratch
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()