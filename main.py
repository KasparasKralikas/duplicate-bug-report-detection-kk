from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from text_preprocessor import clean_text, text_to_padded
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from bug_model import BugModel

oov_token = '<OOV>'
vocab_size = 10000
embedding_dim = 16
training_portion = 0.8
max_length = 800
num_epochs = 30

data_full = pd.read_csv('datasets/training_dataset.csv', sep=',')

# smaller dataset for testing
data_full = data_full[:20000]

data_full['cleaned_description'] = data_full['description'].apply(lambda x: clean_text(x))

training_size = int(len(data_full.index) * training_portion)

training_descriptions = data_full['cleaned_description'][0:training_size]
training_labels = data_full['is_bug'][0:training_size]
testing_descriptions = data_full['cleaned_description'][training_size:]
testing_labels = data_full['is_bug'][training_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_descriptions)
word_index = tokenizer.word_index

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

training_padded = text_to_padded(training_descriptions, tokenizer, max_length)
testing_padded = text_to_padded(testing_descriptions, tokenizer, max_length)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

bug_model = BugModel()

bug_model.constructModel(vocab_size, embedding_dim, max_length)

bug_model.fit_model(training_padded, training_labels, training_padded, training_labels, num_epochs)

bug_model.plot_graphs()

bug_model.save_model()