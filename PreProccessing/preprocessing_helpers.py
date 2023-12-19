from googletrans import Translator
import nltk
import mtranslate

from nltk.corpus import wordnet
# nltk.download('omw-1.4')
# nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import requests

def get_emoji_meaning(word):
            url = f'https://emojipedia.org/{word}'
            res = requests.get(url)
            html = res.content
            soup = BeautifulSoup(html, 'html.parser')
            result = soup.find_all('section', 'description')
            for r in result:
                print(r.text)
franco_to_arabic_dict = {
    'a': 'ا',
    '2': 'أ',
    'b': 'ب',
    't': 'ت',
    'th': 'ث',
    'g': 'ج',
    '7': 'ح',
    '5': 'خ',
    'd': 'ض',
    'r': 'ر',
    'z': 'ز',
    's': 'س',
    '4': 'ش',
    '9': 'ص',
    '6': 'ط',
    'Z': 'ظ',
    '3': 'ع',
    '8': 'غ',
    'f': 'ف',
    'q': 'ق',
    'k': 'ك',
    'l': 'ل',
    'm': 'م',
    'n': 'ن',
    'h': 'ه',
    'w': 'و',
    'y': 'ي'
}

# Print each element on a separate line

def is_laten_word(word):
    pattern = re.compile("^[a-zA-Z0-9]+$")
    match = pattern.match(word)
    if match:
        return True
    else:
        return False


def is_in_english_dictionary(word):
    # Convert the word to lowercase for case-insensitive comparison
    word = word.lower()
    # Check if the word exists in WordNet
    synonym_set = wordnet.synsets(word)
    return len(synonym_set) > 0


def translate_to_arabic(text):
    try:
        # Create a Translator object
        translator = Translator()

        # Translate the text to Arabic
        translation = translator.translate(text, dest='ar')


        # Return the translated text
        return translation.text

    except AttributeError as e:
        print(f"Translation error: {e}")
        return " "
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return " "


def franco_to_arab(word):
    arbic_word=''
    for ch in word:
        if ch in franco_to_arabic_dict:
            arbic_word+=franco_to_arabic_dict[ch]
    return arbic_word

def handle_latin_words(word):
    if is_laten_word(word):
        if(is_in_english_dictionary(word)):
            word=translate_to_arabic(word)
        else:
            word=franco_to_arab(word)
    return word


# print(is_in_english_dictionary("student"))
# print(is_in_english_dictionary("afdl"))

#afdel bernameg llakl
# print(handle_latin_words("afdel"))
# print(handle_latin_words("bernameg"))
# print(handle_latin_words("llakl"))
# print(handle_latin_words("student"))
# print(handle_latin_words("love"))
# print(translate_to_arabic("thes"))
# print(handle_latin_words("المدرسه"))
# print(get_emoji_meaning('❤️'))
