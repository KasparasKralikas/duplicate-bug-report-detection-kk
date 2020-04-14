from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
import re
import string

def clean_text(text, punctuation, stopwords):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return ' '.join(text)

punctuation = string.punctuation
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data_full = pd.read_csv('datasets/training_dataset.csv', sep=',')

# smaller dataset for testing
data_full = data_full[:10]

data_full['cleaned_description'] = data_full['description'].apply(lambda x: clean_text(x, punctuation, stopwords))

descriptions = data_full['cleaned_description'].tolist()

tokenizer = Tokenizer(num_words = 200, oov_token="<OOV>")
tokenizer.fit_on_texts(descriptions)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(descriptions)
padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=800)

print(padded[:10])
