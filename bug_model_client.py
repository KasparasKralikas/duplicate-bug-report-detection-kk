from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from text_preprocessor import clean_text, text_to_padded
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from bug_model import BugModel

class BugModelClient:

    oov_token = '<OOV>'
    vocab_size = 30000
    embedding_dim = 32
    training_portion = 0.8
    max_length = 800
    num_epochs = 50

    data_path = 'datasets/training_dataset.csv'

    data = None

    training_size = None

    word_index = None
    label_index = None

    label_count = 0

    training_padded = None
    training_labels = None
    testing_padded = None
    testing_labels = None

    tokenizer = None
    label_tokenizer = None

    bug_model = BugModel()

    def init_data(self, data_count):
        self.data = pd.read_csv(self.data_path, sep=',')
        self.data = self.data[:data_count]
        self.data['clean_description'] = self.clean_descriptions(self.data['description'])
        self.data['master_id_string'] = self.data['master_id'].apply(lambda x: str(x))
        self.training_size = int(len(self.data.index) * self.training_portion)

        training_descriptions = self.data['clean_description'][0:self.training_size]
        training_labels = self.data['master_id_string'][0:self.training_size]
        testing_descriptions = self.data['clean_description'][self.training_size:]
        testing_labels = self.data['master_id_string'][self.training_size:]

        self.tokenizer = Tokenizer(num_words = self.vocab_size, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(training_descriptions)
        self.word_index = self.tokenizer.word_index
        print(len(self.word_index))

        self.label_tokenizer = Tokenizer(oov_token=self.oov_token)
        self.label_tokenizer.fit_on_texts(training_labels)
        self.label_index = self.label_tokenizer.word_index

        self.label_count = len(self.label_index) + 1

        self.training_padded = np.array(text_to_padded(training_descriptions, self.tokenizer, self.max_length))
        self.training_labels = tf.keras.utils.to_categorical(np.array(self.label_tokenizer.texts_to_sequences(training_labels)))
        self.testing_padded = np.array(text_to_padded(testing_descriptions, self.tokenizer, self.max_length))
        self.testing_labels = tf.keras.utils.to_categorical(np.array(self.label_tokenizer.texts_to_sequences(testing_labels)))

    def clean_descriptions(self, descriptions):
        clean_descriptions = descriptions.apply(lambda x: clean_text(x))
        return clean_descriptions

    def train_model(self):
        self.bug_model.constructModel(self.vocab_size, self.embedding_dim, self.max_length, self.label_count)
        self.bug_model.fit_model(self.training_padded, self.training_labels, self.training_padded, self.training_labels, self.num_epochs)

    def plot_graphs(self):
        self.bug_model.plot_graphs()

    def save_model(self):
        self.bug_model.save_model()

    def load_model(self):
        self.bug_model.load_model()

    def decode_label(self, index):
        reverse_label_index = dict([(value, key) for (key, value) in self.label_index.items()])
        return reverse_label_index.get(index, '?')

    def predict(self, descriptions, n):
        predicting_padded = np.array(text_to_padded(self.clean_descriptions(descriptions), self.tokenizer, self.max_length))
        predictions = self.bug_model.predict(predicting_padded)
        predictions_list = []
        for prediction in predictions:
            prediction_top_n = (-prediction).argsort()[:n]
            prediction_list = []
            for index in prediction_top_n:
                prediction_list.append({
                    'case_id': self.decode_label(index),
                    'probability': prediction[index]
                })
            predictions_list.append(prediction_list)
        return predictions_list

        




    