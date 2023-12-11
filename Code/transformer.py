import tensorflow as tf
import pandas as pd
import sys
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GlobalAveragePooling1D, Dense

sys.path.append('/home/joe/School/Neural/NeuralNetworks-DNN/')  # Replace with the actual path to your project
from PreProccessing import preprocessing as pre

def build_custom_model(input_shape, embedding_dim, num_classes):
    # Input layer
    # inputs = Input(shape=input_shape, dtype=tf.float32, name="input_layer")
    inputs = Input(input_shape, dtype=tf.float32, name="input_layer")

    # Bidirectional LSTM layer
    lstm_layer = Bidirectional(LSTM(units=64, return_sequences=True))(inputs)

    # Global Average Pooling layer
    avg_pooling = GlobalAveragePooling1D()(lstm_layer)

    # Output layer
    outputs = Dense(units=num_classes, activation='softmax')(avg_pooling)

    # Model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='custom_model')

    return model

# Example usage:
input_shape = (39419,15)  # Replace max_sequence_length with the actual length of your sequences
embedding_dim = 15  # Replace with the actual embedding dimension
num_classes = 3

# Build the custom model
custom_model = build_custom_model(input_shape=input_shape, embedding_dim=embedding_dim, num_classes=num_classes)


# Assuming you have your training data and labels
# Replace `x_train` and `y_train` with your actual training data and labels

# Compile the model
custom_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
x_train , y_train , x_test  , y_test = pre.preprocessing()

print(x_train.shape)

custom_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
custom_model.evaluate(x_test, y_test, batch_size=32)
