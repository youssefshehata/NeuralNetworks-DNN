from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
    test_sentences = []
    with open("PreProccessing/sentences.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            cleanSens.append(line.strip())
    with open("PreProccessing/labels.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            labels.append(int(line.strip()))


    with open("PreProccessing/test_sentences.txt", 'r', encoding='utf-8') as file:
        for line in file:
            # Process each line as needed
            test_sentences.append(line.strip())
    return cleanSens , labels  , test_sentences


def tokenize(max_sequence_length  = 36):
    # getting data from files and creatig new dataframe
    cleanSens , labels , test_sentences = importingData()
    columns = ['reviews', 'rating']
    df = pd.DataFrame(columns=columns)
    df['reviews'] = cleanSens
    df['rating'] = labels

    test = pd.DataFrame(columns=['test'])
    test['test'] = test_sentences

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
    test_sentences= test.test.to_numpy()






    # Tokenization
    tokenizer = Tokenizer(num_words=num_unique_words)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    test_sentences = tokenizer.texts_to_sequences(test_sentences)
    index_word = tokenizer.index_word

    # Padding Sequences

    train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    val_padded = pad_sequences(val_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    test_sentences_padded =  pad_sequences(test_sentences, maxlen=max_sequence_length, padding='post', truncating='post')


    # print(train_padded[9158])
    # reversed index ,word dictionary to get the word of each index
    # reverse_words_index = dict([(value, key) for (key, value) in word_index.items()])


    return num_unique_words , max_sequence_length , train_padded , train_labels , val_padded , val_labels ,cleanSens , test_sentences_padded



