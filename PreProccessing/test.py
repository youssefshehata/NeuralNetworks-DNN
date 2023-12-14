import re
import nltk
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



def steming(text):
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

# Example usage
arabic_text = "التحليللللل اللغوي للغة العربية رائععع نيك وكسممم ززززز .... ابن زاايد"
processed_text = preprocess_arabic_text(arabic_text)
print(processed_text)
