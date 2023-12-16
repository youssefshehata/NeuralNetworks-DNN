
import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath("/home/joe/School/Neural/NeuralNetworks-DNN/")))
from keras.layers import Embedding , LSTM , Dense
from keras.models import Sequential
import keras



import numpy as np



def mylstm( num_unique_words, max_sequence_length , train_padded , train_labels , val_padded , val_labels):
    




    # creating model
    model = Sequential()
    model.add(Embedding(num_unique_words+1, 100, input_length=max_sequence_length , mask_zero=True))
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
    print("Actual labels : ",train_labels[10:20])    
    print("Predicted labels : ",predictions[10:20])
