import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath("/home/joe/School/Neural/NeuralNetworks-DNN/")))
import PreProccessing.Embedding as embedding
import PreProccessing.Tokenization as tokenizer
from Code.transformer import trans

from Code.LSTM import mylstm
from keras.layers import Embedding , LSTM , Dense
from keras.models import Sequential
import keras
import numpy as np

def label_encoder(label):
    if label == "-1":
        return 0
    elif label == "0":
        return 1
    else:
        return 2
    
def label_decoder(label):
    if label == "0":
        return -1
    elif label == "1":
        return 0
    else:
        return 1






def main():
    # this block includes the code that actually works (only lstm works ) , comment everything else to try it out
    ################################################################

    num_unique_words , max_sequence_length , train_padded , train_labels , val_padded , val_labels,sentences = tokenizer.tokenize()

    # lstm code
    mylstm( num_unique_words, max_sequence_length , train_padded , train_labels , val_padded , val_labels)

    
    # transformer code 
    # for i in range(len(train_labels)):
    #     train_labels[i] = label_encoder(train_labels[i])
    # for i in range(len(val_labels)):
    #     val_labels[i] = label_encoder(val_labels[i])
    # trans(max_sequence_length, num_unique_words, 3 , train_padded , val_padded , train_labels , val_labels )


    ###############################################################

    


    # num_unique_words , max_sequence_length , train_padded , train_labels , val_padded , val_labels , data = embedding.embedding()

    

main()