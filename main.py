import tensorflow as tf
import keras
from keras.layers import Embedding,LSTM, Dense 
from keras.models import Sequential

import Code.LSTM as myLSTM
import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath("/home/joe/School/Neural/NeuralNetworks-DNN/")))

import PreProccessing.Tokenization as tokenizer


def main():
    num_unique_words, max_sequence_length , train_padded , train_labels , val_padded , val_labels = tokenizer.tokenize(max_sequence_length= 36)# using the average length of the sentences
    
    
    
    
    # creating model
    model = Sequential()
    model.add(Embedding(num_unique_words+1, 100, input_length=max_sequence_length , mask_zero=True))
    # model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.1))

   
   
   
   
    model.add(Dense(1, activation='sigmoid'))
    
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim=keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy'])
    model.fit(train_padded, train_labels, epochs=6, validation_data=(val_padded, val_labels), verbose=2)
    score = model.evaluate(val_padded, val_labels, verbose=2)


    print(f"Test Accuracy:", score[1])
    predictions = model.predict(train_padded)
    predictions = [-1 if P < 0.33 else (0 if P < 0.67 else 1) for P in predictions]

    print(train_labels[10:20])    
    print(predictions[10:20])






main()

# vocab_size, embedding_dim, max_sequence_length, embedding_matrix , x_train , y_train , x_test , y_test ,labels= emb.embedding()
# # Input layer
# inputs = embedding_matrix

# # Placeholder for labels


# # Embedding layer
# embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length, weights=[embedding_matrix], trainable=False)(inputs)

# # Custom LSTM layer
# custom_lstm_layer = LSTM.myLstm(units=128)(embedding_layer)

# # Extract the hidden state from the custom LSTM layer
# lstm_output = custom_lstm_layer[0]

# # Dense layer with sigmoid activation for binary classification
# output = Dense(1, activation='sigmoid')(lstm_output)

# # Binary cross-entropy loss
# loss = tf.keras.losses.BinaryCrossentropy()(labels, output)

# # Adam optimizer
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# # Minimize the loss
# train_op = optimizer.minimize(loss)

# # TensorFlow Session
# with tf.Session() as sess:
#     # Initialize variables
#     sess.run(tf.global_variables_initializer())
#     epochs = 3
#     # Training loop (replace x_train and y_train with your actual training data)
#     for epoch in range(epochs):
#         _, epoch_loss = sess.run([train_op, loss], feed_dict={inputs: x_train, labels: y_train})

#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")

#     # Evaluation (replace x_test and y_test with your actual test data)
#     test_loss = sess.run(loss, feed_dict={inputs: x_test, labels: y_test})
#     print(f"Test Loss: {test_loss}")
