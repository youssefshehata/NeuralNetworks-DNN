import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath("/home/joe/School/Neural/NeuralNetworks-DNN/")))
from keras.layers import Embedding 
import PreProccessing.Tokenization as tokenizer




import numpy as np

from Code.LSTM import LstmParam, LstmNetwork

class ToyLabeledLossLayer:
    """
    Computes square loss with the first element of hidden layer array.
    Assumes labeled data with a target label.
    """
    @classmethod
    def loss(cls, pred, label):
        return (pred - label) ** 2

    @classmethod
    def bottom_diff(cls, pred, label):
        diff = np.zeros_like(pred)
        diff = 2 * (pred - label)
        return diff


def labeled_example():
    num_unique_words, max_sequence_length , train_padded , train_labels , val_padded , val_labels = tokenizer.tokenize(max_sequence_length= 36)# using the average length of the sentences
    # emb = Embedding(num_unique_words+1, 15, input_length=max_sequence_length , mask_zero=True)
    # train_padded = emb(train_padded)
    np.random.seed(0)

    # Parameters for input data dimension and LSTM cell count
    mem_cell_ct = 100
    x_dim = 36
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    

    num_epochs = 2

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        for i,sentence in enumerate(train_padded):
            input_val = sentence
            label = train_labels[i]

            lstm_net.x_list_add(input_val)

            # Get the predicted value from the LSTM network
            predicted_value = lstm_net.lstm_node_list[-1].state.h[0]
            if np.abs(predicted_value - label) < 0.5:
                correct_predictions += 1

            # Compute the loss using the labeled data
            loss = ToyLabeledLossLayer.loss(predicted_value, label)
            total_loss += loss

            # Backpropagation
            diff_h = ToyLabeledLossLayer.bottom_diff(predicted_value, label)
            diff_s = np.zeros(mem_cell_ct)
            lstm_net.lstm_node_list[-1].top_diff_is(diff_h, diff_s)

            # Apply gradients
            lstm_param.apply_diff(lr=0.1)

            # Clear input sequence for the next iteration
            lstm_net.x_list_clear()
        accuracy = correct_predictions / len(train_labels)


        # Print average loss for the epoch
        average_loss = total_loss / len(train_labels)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss} , Accuracy: {accuracy}")

    # After training, you can use the trained LSTM network to predict labels for new sequences
    correct_predictions = 0
    for i , sentence in enumerate(val_padded):
        input_val = sentence
        label = val_labels[i]
        lstm_net.x_list_add(input_val)
        predicted_value = lstm_net.lstm_node_list[-1].state.h[0]
        if np.abs(predicted_value - label) < 0.5:
            correct_predictions += 1
        lstm_net.x_list_clear()
        print(f"Predicted Label for New Sequence: {predicted_value} , Actual Label: {val_labels[i]}")

    accuracy = correct_predictions / len(val_labels)
    print(f"Accuracy on New Sequences: {accuracy}")
if __name__ == "__main__":
    labeled_example()

















# def main():

#     num_unique_words, max_sequence_length , train_padded , train_labels , val_padded , val_labels = tokenizer.tokenize(max_sequence_length= 36)# using the average length of the sentences
    




    # # creating model
    # model = Sequential()
    # model.add(Embedding(num_unique_words+1, 100, input_length=max_sequence_length , mask_zero=True))
    # model.add(LSTM(128, dropout=0.1))
    # model.add(Dense(1, activation='sigmoid'))
    # loss = keras.losses.BinaryCrossentropy(from_logits=False)
    # optim=keras.optimizers.Adam(learning_rate=0.001)
    # model.compile(loss=loss, optimizer=optim, metrics=['accuracy'])
    # model.fit(train_padded, train_labels, epochs=6, validation_data=(val_padded, val_labels), verbose=2)
    # score = model.evaluate(val_padded, val_labels, verbose=2)


    # print(f"Test Accuracy:", score[1])
    # predictions = model.predict(train_padded)
    # predictions = [-1 if P < 0.33 else (0 if P < 0.67 else 1) for P in predictions]

    # print(train_labels[10:20])    
    # print(predictions[10:20])






# main()

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
