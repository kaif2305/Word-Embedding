from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Load Datasey
vocab_size = 10000
max_len = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Decode reviews to text for preprocessing
# Load the word-to-index dictionary from the IMDb dataset.
# Each word is mapped to a unique integer index (e.g., 'great': 456).
word_index = imdb.get_word_index()

# Create a reverse dictionary: index → word.
# This allows us to convert integer-encoded reviews back to words.
reverse_word_index = {value: key for key, value in word_index.items()}

# Decode the first 5 reviews from integer format to plain English text.
# For each review in the first 5 reviews of the training set:
#   - For each integer in the review:
#       - Subtract 3 because the first 3 indices are reserved:
#           0 = padding, 1 = start of sequence, 2 = unknown word.
#       - Get the corresponding word from reverse_word_index.
#       - If the index is not found in the dictionary, return "?".
#   - Join all the decoded words with a space to form the final readable review.
decoded_reviews = [
    " ".join([reverse_word_index.get(i - 3, "?") for i in review])
    for review in X_train[:5]
]


# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

''' Load GloVe embeddings'''

embedding_index = {}
glove_file = "glove.6B/glove.6B.100d.txt"

# Open the GloVe file in read mode using UTF-8 encoding
with open(glove_file, "r", encoding='utf-8') as file:
    
    # Loop through each line in the file
    for line in file:
        # Split the line into parts: first is the word, the rest are the 100-dimensional numbers
        values = line.split()
        
        # The first element is the word itself (e.g., "apple")
        word = values[0]
        
        # The remaining elements are the word's embedding coefficients (as strings)
        # Convert them to a NumPy array of float32 type for efficient numerical operations
        coefs = np.asarray(values[1:], dtype="float32")
        
        # Add the word and its corresponding vector to the embedding dictionary
        embedding_index[word] = coefs


print(f"Loaded {len(embedding_index)} word vectors")

''' Prepare embedding matrix '''

# Since we are using 'glove.6B.100d.txt', each word is represented by a 100-dimensional vector.
embedding_dim = 100

# Initialize the embedding matrix with all zeros.
# Shape: (vocab_size, embedding_dim)
# - Each row corresponds to a word index (from 0 to vocab_size - 1)
# - Each row will later be filled with the corresponding GloVe vector if available
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Loop through each word and its corresponding index in the dataset's vocabulary
for word, i in word_index.items():

    # Only consider the top 'vocab_size' most frequent words.
    # Words with index >= vocab_size will be ignored (e.g., rare words)
    if i < vocab_size:

        # Get the GloVe vector for the current word, if it exists in the pre-trained embeddings
        embedding_vector = embedding_index.get(word)

        # If a GloVe vector is found for the word
        if embedding_vector is not None:
            
            # Assign the GloVe vector to the corresponding row (index 'i') in the embedding matrix
            # This means: when our model sees word index 'i', it will use this pre-trained GloVe vector
            embedding_matrix[i] = embedding_vector


# Define LSTM model with GloVe embeddings
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)


lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

lstm_history = lstm_model.fit(
    X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)
print(f"LSTM without GloVe Test Accuracy: {lstm_accuracy:.4f}")
print(f"LSTM model with GloVe Test Accuracy: {accuracy:.4f}")

import matplotlib.pyplot as plt

# Plot accuracy comparison
models = ['LSTM', 'LSTM GloVe']
accuracies = [lstm_accuracy, accuracy]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Comparison of LSTM with and without word embeddings')
plt.ylabel('Accuracy')
plt.show()
