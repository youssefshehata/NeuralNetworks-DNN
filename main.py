import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense
import Code.LSTM as LSTM
import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath("/home/joe/School/Neural/NeuralNetworks-DNN/")))

import PreProccessing.Embedding as emb





vocab_size, embedding_dim, max_sequence_length, embedding_matrix , x_train , y_train , x_test , y_test ,labels= emb.embedding()
# Input layer
inputs = embedding_matrix

# Placeholder for labels


# Embedding layer
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length, weights=[embedding_matrix], trainable=False)(inputs)

# Custom LSTM layer
custom_lstm_layer = LSTM.myLstm(units=128)(embedding_layer)

# Extract the hidden state from the custom LSTM layer
lstm_output = custom_lstm_layer[0]

# Dense layer with sigmoid activation for binary classification
output = Dense(1, activation='sigmoid')(lstm_output)

# Binary cross-entropy loss
loss = tf.keras.losses.BinaryCrossentropy()(labels, output)

# Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Minimize the loss
train_op = optimizer.minimize(loss)

# TensorFlow Session
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    epochs = 3
    # Training loop (replace x_train and y_train with your actual training data)
    for epoch in range(epochs):
        _, epoch_loss = sess.run([train_op, loss], feed_dict={inputs: x_train, labels: y_train})

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")

    # Evaluation (replace x_test and y_test with your actual test data)
    test_loss = sess.run(loss, feed_dict={inputs: x_test, labels: y_test})
    print(f"Test Loss: {test_loss}")
