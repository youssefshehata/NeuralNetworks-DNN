import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example: List of sentences (sequences) with word embeddings
sequences = [
    np.random.rand(5, 15),  # Sequence 1 with 5 words
    np.random.rand(8, 15),  # Sequence 2 with 8 words
    np.random.rand(6, 15),  # Sequence 3 with 6 words
]

# Padding sequences to a   length
max_sequence_length = 8  # Assuming a maximum length of 8 for illustration
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Convert to NumPy array
input_data = np.array(padded_sequences)

# Display the input data
print("Original Sequences:")
for sequence in sequences:
    print(sequence)
    break

