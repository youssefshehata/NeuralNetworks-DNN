from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import fasttext
import numpy as np
import re
from keras.layers import Embedding, LSTM, Dense , Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def embedding():
    cleanSens = []
    labels = []
    max = 0
    with open("PreProccessing/sentences.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            if len(line.strip().split()) > max:
                max = len(line.strip().split())
            cleanSens.append(line.strip())
    with open("PreProccessing/labels.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            labels.append(int(line.strip()))

    x_train, x_test, y_train, y_test = train_test_split(cleanSens, labels, test_size=0.2, random_state=42)




    # TOKENIZATION 
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    vocab_size = len(tokenizer.word_index) + 1


    # PADDING 
    max_sequence_length = 50# Adjust as needed
    x_train = pad_sequences(x_train , padding = "post", maxlen=max_sequence_length)
    x_test = pad_sequences(x_test,  padding = "post", maxlen=max_sequence_length)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)



    embedding_dim = 300

    arabic_fasttext_model_path = 'PreProccessing/cc.arz.300.bin'  # Replace with the path to the pre-trained model file
    model = fasttext.load_model(arabic_fasttext_model_path)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = model[word]

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
                                
    return  vocab_size, embedding_dim, max_sequence_length, embedding_matrix , x_train , y_train , x_test , y_test , labels

    # embedding_layer =Embedding(vocab_size, embedding_dim,weights=[embedding_matrix], input_length=max_sequence_length,trainable=False)
    # # create model
    # model = Sequential()
    # model.add(embedding_layer)
    # model.add(LSTM(128))
    # # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))   
    # model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    # history = model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.2)


    # score = model.evaluate(x_test, y_test, verbose=1)
    # print(f"Test Score:", score[0])
    # print(f"Test Accuracy:", score[1])

    # # # Make predictions on the test set
    # # y_pred = model.predict(x_test)

    # # # If your model outputs probabilities, you might want to threshold them to get binary predictions
    # # threshold = 0.5  # Adjust as needed
    # # y_pred_binary = (y_pred > threshold).astype(int)



