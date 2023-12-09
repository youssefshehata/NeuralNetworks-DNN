import pandas as pd
import nltk
import csv
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import re
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


# def reverse_words_in_array(array):
#     reversed_array = []
#     for string in array:
#         reversed_string = ' '.join(word[::-1] for word in string.split())
#         reversed_array.append(reversed_string)
#     return reversed_array
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit_transform(train_fit)
    return vector

Dataset = pd.read_excel("/home/joe/School/Neural/NeuralNetworks-DNN/Data/train.xlsx")
# Dataset = pd.read_csv('/home/joe/School/Neural/NeuralNetworks-DNN/Data/train.csv' , encoding='utf-8')

Dataset = Dataset[['review_description']]

pattern = r'[^\u0600-\u06FF\s]+'    
clean_data_list = []
stop_words = set(stopwords.words('arabic'))
for i, row in Dataset.iterrows():

    content = row['review_description']
    content_clean = re.sub(pattern, '', content)

    tokens = [re.sub(r'[^\w\s]', '', word) for sent in nltk.sent_tokenize(content_clean) for word in
              nltk.word_tokenize(sent) if
              word.casefold() not in stop_words and word.casefold() not in string.punctuation and len(word) > 1]
    # tokens = reverse_words_in_array(tokens)

    if len(tokens) > 0 and tokens != [' ']:
        clean_data_list.append(tokens)



# Example Arabic dataset (list of lists)
arabic_dataset = clean_data_list


# Flatten the dataset to create a list of words
flat_arabic_dataset = [word for sentence in arabic_dataset for word in sentence]

# Create a vocabulary (unique words) and assign indices to each word
vocab_size = len(set(flat_arabic_dataset))
word_to_index = {word: idx + 1 for idx, word in enumerate(set(flat_arabic_dataset))}

# Define the maximum length of a sentence (assuming a fixed length for simplicity)
max_sentence_length = max(len(sentence) for sentence in arabic_dataset)

# Convert each word in the dataset to its corresponding index
indexed_dataset = [[word_to_index[word] for word in sentence] for sentence in arabic_dataset]

# Define the embedding layer with character embeddings
embedding_dim = 15  # You can adjust the embedding dimension
model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_sentence_length))
model.add(Flatten())

# Compile the model (not training in this example)
model.compile(optimizer='adam', loss='mse')


# padding the dataset to have the same length
padded_dataset = pad_sequences(indexed_dataset, maxlen=295, padding='post', truncating='post')


embedded_representation = model.predict(np.array(padded_dataset))


# Print the learned character embeddings for each word
embedding_weights = model.layers[0].get_weights()[0]
# print("\nLearned Character Embeddings:")
# for word, index in word_to_index.items():
#     print(f"{word}: {embedding_weights[index]}")
