import pandas as pd
import preprocessing_helpers as hp
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from concurrent.futures import ThreadPoolExecutor


# nltk.download('punkt')
# nltk.download('stopwords')


def remove_consecutive_repeated_letters(word):
    # Use regular expression to remove consecutive repeated letters
    word = re.sub(r'(.)\1+', r'\1', word)
    if len(word) > 1:
        return word
    return ''


def stemming(text):
    # Remove non-Arabic characters
    text = re.sub(r'[^ุก-ู\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('arabic'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    tokens = [remove_consecutive_repeated_letters(word) for word in tokens]

    # Stemming using SnowballStemmer
    stemmer = SnowballStemmer('arabic')
    tokens = [stemmer.stem(word) for word in tokens]

    # Join the tokens back into a single string
    processed_text = ' '.join(tokens)

    return processed_text


def removeNulls(sentences, lebels):
    newSens = []
    newLabels = []
    for i in range(len(sentences)):
        if len(sentences[i]) > 1:
            newSens.append(sentences[i])
            if (len(lebels) > 0):
                newLabels.append(lebels[i])
    return newSens, newLabels


def readData():
    data = pd.read_csv("C:/Users/saif_/PycharmProjects/NeuralNetworks-DNN/Data/test _no_label.csv")
    data.dropna()
    data = data[data['review_description'].apply(len) > 1]

    sentences = data[['review_description']]

    return sentences


def readData_train():
    data = pd.read_excel("C:/Users/saif_/PycharmProjects/NeuralNetworks-DNN/Data/train.xlsx")
    data.dropna()
    data = data[data['review_description'].apply(len) > 1]
    labels=data[['rating']]
    sentences = data[['review_description']]

    return sentences , labels


def preprocessing():
    data = pd.read_excel("C:/Users/saif_/PycharmProjects/NeuralNetworks-DNN/Data/train.xlsx", sheet_name='Sheet1')
    data.dropna()
    data = data[data['review_description'].apply(len) > 1]

    sentences = data[['review_description']]
    labels = data[['rating']].values
    pattern = r'[^\u0600-\u06FF\s]+'
    cleanSens = []
    stop_words = set(nltk.corpus.stopwords.words('arabic'))

    batch_size = 1000  # Experiment with different batch sizes

    for batch_start in range(0, len(sentences), batch_size):
        batch_end = min(batch_start + batch_size, len(sentences))
        batch = sentences.iloc[batch_start:batch_end]

        batch_cleanSens = []  # Accumulator for processed sentences in the current batch

        for i, row in batch.iterrows():
            content = row['review_description']
            tokens = [hp.handle_latin_words(re.sub(r'[^\w\s]', '', word)) for word in word_tokenize(content) if
                      word.casefold() not in stop_words and word.casefold() not in string.punctuation and len(word) > 1]

            sentence = ' '.join(tokens)
            print(sentence)
            batch_cleanSens.append(sentence)

        cleanSens.extend(batch_cleanSens)

    cleanSens, labels = removeNulls(cleanSens, labels)

    with open("sentences.txt", 'w', encoding='utf-8') as file:
        for sentence in cleanSens:
            file.write(sentence + '\n')

    with open("labels.txt", 'w', encoding='utf-8') as file:
       for label in labels:
           file.write(str(label[0]) + '\n')
def removeNulls(cleanSens, labels):
    # Implement your logic to remove nulls or handle missing data
    return cleanSens, labels


# write sens and labels to files


def preprocessingTest():
    sentences = readData()
    pattern = r'[^\u0600-\u06FF\s]+'
    cleanSens = []
    stop_words = set(nltk.corpus.stopwords.words('arabic'))
    for i, row in sentences.iterrows():
        content = row['review_description']
        # content_clean = re.sub(pattern, '', content)
        tokens = [hp.handle_latin_words(re.sub(r'[^\w\s]', '', word)) for word in word_tokenize(content) if
                  word.casefold() not in stop_words and word.casefold() not in string.punctuation and len(word) > 1]

        #tokens = [stemming(word) for word in tokens]
        sentence = ' '.join(tokens)
        cleanSens.append(sentence)

    # Open the file in write mode with UTF-8 encoding
    with open("Testsentences.txt", 'w', encoding='utf-8') as file:
        # Write each sentence to the file
        for sentence in cleanSens:
            file.write(sentence + '\n')


#preprocessingTest()
preprocessing()