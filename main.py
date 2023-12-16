import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath("/home/joe/School/Neural/NeuralNetworks-DNN/")))
import PreProccessing.Embedding as embedding
import PreProccessing.Tokenization as tokenizer
# from Code.transformer import trans

from Code.LSTM import mylstm
from keras.layers import Embedding , LSTM , Dense
from keras.models import Sequential
import keras
import numpy as np





def main():
    num_unique_words , max_sequence_length , train_padded , train_labels , val_padded , val_labels , data = embedding.embedding()



    # this block includes the code that actually works (only lstm works ) , comment everything else to try it out
    ################################################################

    # num_unique_words , max_sequence_length , train_padded , train_labels , val_padded , val_labels,sentences = tokenizer.tokenize()

    # lstm code
    # mylstm( num_unique_words, max_sequence_length , train_padded , train_labels , val_padded , val_labels)
    # transformer code 
    # trans(max_seq_length, vocab_size, num_classes , train_padded , val_padded , train_labels , val_labels ):


    ###############################################################

    
    
    
    
    vocab_size, embedding_dim = data.shape[1], data.shape[0]

    print(data.shape)
    # (24202, 14916)
    # creating model
    model = Sequential()
    # model.add(Embedding(30253,16723,weights=[data] ,input_length=max_sequence_length  ,trainable=False))
    model.add(Dense(units=embedding_dim,input_dim=vocab_size,weights=[data.T, np.zeros(vocab_size)] , trainable=False))
    model.add(LSTM(128, dropout=0.1))
    model.add(Dense(1, activation='sigmoid'))
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim=keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy'])
    print(train_padded.shape)
    # (2, 177937)

    model.fit(train_padded, train_labels, epochs=6, validation_data=(val_padded, val_labels), verbose=2)
    score = model.evaluate(val_padded, val_labels, verbose=2)

    
    print(f"Test Accuracy:", score[1])
    predictions = model.predict(train_padded)
    predictions = [-1 if P < 0.33 else (0 if P < 0.67 else 1) for P in predictions]

    print(train_labels[10:20])    
    print(predictions[10:20])

main()