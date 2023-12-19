import sys 
import os
from keras.layers import Bidirectional
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import keras
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath("NeuralNetworks-DNN/")))





def mylstm(num_unique_words, max_sequence_length, train_padded, train_labels, val_padded, val_labels , test_sentences):
  
    # print(train_padded.shape)
    # print(test_sentences.shape)

    # print(train_padded[0:3])
    # print(test_sentences[0:3])
    
    
    # # Creating model
    model = Sequential()



    model.add(Embedding(num_unique_words + 1 , 200, input_length=max_sequence_length, mask_zero=True))

    model.add(Bidirectional(LSTM(128, dropout=0.1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    loss = keras.losses.sparse_categorical_crossentropy
    optim = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy'])
    model.fit(train_padded, train_labels, epochs=3, validation_data=(val_padded, val_labels), verbose=2)

    score = model.evaluate(val_padded, val_labels, verbose=2)
    print(f"Test Accuracy:", score[1])

    predictions = model.predict(test_sentences)
    predictions = np.argmax(predictions, axis=1) - 1



    # print("Actual labels : ", train_labels[10:20])
    # print("Predicted labels : ", predictions[10:20])



    data = pd.read_csv("C:/Users/saif_/PycharmProjects/NeuralNetworks-DNN/Data/test _no_label.csv")
    submimssion= pd.DataFrame()
    submimssion["ID"] = data['ID']
    submimssion["rating"] = predictions
    submimssion.to_csv("LSTMsubmission.csv", index=False)




# Example usage:
# mylstm(num_unique_words, max_sequence_length, train_padded, train_labels, val_padded, val_labels)
