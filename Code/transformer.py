import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding ,  GlobalAveragePooling1D, Dense
import transformers as Transformer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Assume you have your embedded dataset (embedded_representation) and labels (y_labels)
# Example data (replace this with your actual data)
embedded_representation = np.random.rand(100, 10, 20)  # 100 samples, 10 words per sample, 20 embedding dimensions
y_labels = np.random.randint(2, size=(100,))

# Define the Transformer model
def build_transformer_model(input_shape, embedding_dim, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = Embedding(input_dim=embedding_dim, output_dim=embedding_dim)(inputs)
    for _ in range(num_transformer_blocks):
        x = Transformer(num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)(x)
    x = GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs)

# Set hyperparameters
embedding_dim = 20  # Change this based on the dimensionality of your embeddings
num_heads = 2
ff_dim = 32
num_transformer_blocks = 2
mlp_units = [128]

# Build and compile the model
transformer_model = build_transformer_model((None,), embedding_dim, num_heads, ff_dim, num_transformer_blocks, mlp_units)
transformer_model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Print a summary of the model architecture
transformer_model.summary()

# Train the model
transformer_model.fit(embedded_representation, y_labels, epochs=10, batch_size=32, validation_split=0.2)
