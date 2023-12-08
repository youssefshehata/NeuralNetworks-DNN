import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import re
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit_transform(train_fit)
    return vector


Dataset = pd.read_excel("D:\School\NeuralNetworks-DNN\Data\train.xlsx")
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
    clean_data_list.append(tokens)

print(clean_data_list)
