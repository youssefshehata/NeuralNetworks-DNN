from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from collections import Counter





def decode(reverse_words_index,text): 
    return ' '.join([reverse_words_index.get(i, '') for i in text])

def couter_word(txt):
    count = Counter()
    for i in txt:
        for word in i.split():
            count[word] += 1
    return count

def importingData():
    cleanSens = []
    labels = []
    with open("PreProccessing/sentences.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            cleanSens.append(line.strip())
    with open("PreProccessing/labels.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            labels.append(int(line.strip()))
    return cleanSens , labels 


def tokenize(max_sequence_length  = 20):
    # getting data from files and creatig new dataframe
    cleanSens , labels = importingData()
    columns = ['reviews', 'rating']
    df = pd.DataFrame(columns=columns)
    df['reviews'] = cleanSens
    df['rating'] = labels
    average_length = df['reviews'].apply(len).mean()
    # print(average_length)


    # counting unique words
    counter = couter_word(cleanSens)
    num_unique_words = len(counter)



    # splitting data into training and validation sets
    train_size = int(len(cleanSens) * .8)
    train_df = df[:train_size]
    val_df = df[train_size:]

    train_sentences = train_df.reviews.to_numpy()
    train_labels = train_df.rating.to_numpy()
    val_sentences = val_df.reviews.to_numpy()
    val_labels = val_df.rating.to_numpy()





    # Tokenization
    tokenizer = Tokenizer(num_words=num_unique_words)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)


    # Padding Sequences
    
    train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    val_padded = pad_sequences(val_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    # print(train_padded[9158])
    # reversed index ,word dictionary to get the word of each index
    # reverse_words_index = dict([(value, key) for (key, value) in word_index.items()])


    return num_unique_words , max_sequence_length , train_padded , train_labels , val_padded , val_labels




    # # TOKENIZATION 
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(x_train)

    # x_train = tokenizer.texts_to_sequences(x_train)
    # x_test = tokenizer.texts_to_sequences(x_test)

    # vocab_size = len(tokenizer.word_index) + 1


    # # PADDING 
    # max_sequence_length = 50# Adjust as needed
    # x_train = pad_sequences(x_train , padding = "post", maxlen=max_sequence_length)
    # x_test = pad_sequences(x_test,  padding = "post", maxlen=max_sequence_length)

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)



    # embedding_dim = 300

    # arabic_fasttext_model_path = 'PreProccessing/cc.arz.300.bin'  # Replace with the path to the pre-trained model file
    # model = fasttext.load_model(arabic_fasttext_model_path)

    # embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # for word, i in tokenizer.word_index.items():
    #     embedding_vector = model[word]

    #     if embedding_vector is not None:
    #         embedding_matrix[i] = embedding_vector
                                




    # embedding_layer =Embedding(vocab_size, embedding_dim,weights=[embedding_matrix], input_length=max_sequence_length,trainable=False)
    # # create model
    # model = Sequential()
    # model.add(embedding_layer)
    # model.add(SimpleRNN(128 , activation='relu',return_sequences=False))
    # # model.add(Dropout(0.5))
    # model.add(Dense(10))
    # loss = keras.losses.BinaryCrossentropy()    
    # optim = keras.optimizers.Adam(learning_rate=0.001)

    # model.compile( loss=loss , optimizer=optim , metrics=['accuracy'])



    # model.fit(x_train, y_train, batch_size=64, epochs=6, verbose = 2,validation_split=0.2)


    # score = model.evaluate(x_test, y_test, verbose=2    )
    # print(f"Test Score:", score[0])
    # print(f"Test Accuracy:", score[1])
    # return  vocab_size, embedding_dim, max_sequence_length, embedding_matrix , x_train , y_train , x_test , y_test , labels


    # # # Make predictions on the test set
    # # y_pred = model.predict(x_test)

    # # # If your model outputs probabilities, you might want to threshold them to get binary predictions
    # # threshold = 0.5  # Adjust as needed
    # # y_pred_binary = (y_pred > threshold).astype(int)

# def encode_labels(labels):
#     for label in labels:
#         if label == -1:
#             label = 0
#         elif label == 0:
#             label = 1
#         else:
#             label = 2
#     return labels 


# def encode_labels(labels):
#     for label in labels:
#         if label == 0:
#             label = -1
#         elif label == 1:
#             label = 0
#         else:
#             label = 1
#     return labels

