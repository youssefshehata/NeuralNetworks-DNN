import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

def remove_consecutive_repeated_letters(word):
    # Use regular expression to remove consecutive repeated letters
    word =  re.sub(r'(.)\1+', r'\1', word)
    if len(word) > 1:
        return word
    return ''



def stemming(text):
    # Remove non-Arabic characters
    text = re.sub(r'[^ء-ي\s]', '', text)

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

def removeNulls(sentences ,  lebels):
    newSens= []
    newLabels = []
    for i in range(len(sentences)):
        if len(sentences[i]) > 1:
            newSens.append(sentences[i])
            newLabels.append(lebels[i])
    return newSens , newLabels


def preprocessing():
    data = pd.read_excel("/home/joe/School/Neural/NeuralNetworks-DNN/Data/train.xlsx", sheet_name='Sheet1')
    data.dropna()
    data = data[data['review_description'].apply(len) > 1]

    sentences = data[['review_description']]
    labels = data[['rating']].values
    pattern = r'[^\u0600-\u06FF\s]+'    
    cleanSens = []
    stop_words = set(nltk.corpus.stopwords.words('arabic'))
    for i, row in sentences.iterrows():
        content = row['review_description']
        content_clean = re.sub(pattern, '', content)
        tokens = [re.sub(r'[^\w\s]', '', word) for word in nltk.word_tokenize(content_clean) if
                word.casefold() not in stop_words and word.casefold() not in string.punctuation and len(word) > 1]
        
        sentence = ' '.join(tokens)
        sentence = stemming(sentence)

        cleanSens.append(sentence)

    cleanSens , labels = removeNulls(cleanSens , labels)
    
