import pandas as pd
import nltk
import re
import string

def preprocessing():
    data = pd.read_excel("/home/joe/School/Neural/NeuralNetworks-DNN/Data/train.xlsx", sheet_name='Sheet1')
    data.dropna()
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

        cleanSens.append(sentence)

    with open('sentences.txt', 'w', encoding='utf-8') as file:
        for sentence in cleanSens:
            file.write(sentence + '\n')
    with open('labels.txt', 'w', encoding='utf-8') as file:
        for label in labels:
            file.write(str(label[0]) + '\n')






