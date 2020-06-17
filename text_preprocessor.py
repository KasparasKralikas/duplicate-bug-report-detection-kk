import nltk
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

punctuation = string.punctuation
stopwords = nltk.corpus.stopwords.words('english')
wnl = nltk.stem.WordNetLemmatizer()

trunctating_type='post'
padding_type='post'

def clean_text(text):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in punctuation])
    tokens = re.split('\W+', text)
    #text = [wnl.lemmatize(word) for word in tokens if word not in stopwords]
    text = [word for word in tokens if word not in stopwords]
    return ' '.join(text)

def text_to_padded(text, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, padding=padding_type, truncating=trunctating_type, maxlen=max_length)
    return padded

