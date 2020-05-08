from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from text_preprocessor import clean_text, text_to_padded
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from bug_model import BugModel

class BugModelClient:

    oov_token = '<OOV>'
    vocab_size = 60000
    embedding_dim = 32
    training_portion = 0.8
    max_length = 800
    num_epochs = 3
    dropout = 0.1

    data_path = 'datasets/training_dataset_pairs.csv'

    data = None

    training_size = None

    word_index = None

    tokenizer = None

    bug_model = BugModel()

    def init_data(self, data_count):
        self.data = pd.read_csv(self.data_path, sep=',')
        self.data = self.data[:data_count]
        self.data['clean_description_1'] = self.clean_descriptions(self.data['description_1'])
        self.data['clean_description_2'] = self.clean_descriptions(self.data['description_2'])
        self.training_size = int(len(self.data.index) * self.training_portion)

        X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(self.data['clean_description_1'], self.data['clean_description_2'], self.data['duplicates'], test_size=0.2)

        self.tokenizer = Tokenizer(num_words = self.vocab_size, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(X1_train)
        self.tokenizer.fit_on_texts(X2_train)
        self.word_index = self.tokenizer.word_index
        print(len(self.word_index))

        X1_train = np.array(text_to_padded(X1_train, self.tokenizer, self.max_length))
        X1_test = np.array(text_to_padded(X1_test, self.tokenizer, self.max_length))
        X2_train = np.array(text_to_padded(X2_train, self.tokenizer, self.max_length))
        X2_test = np.array(text_to_padded(X2_test, self.tokenizer, self.max_length))
        
        self.X1_train = X1_train
        self.X1_test = X1_test
        self.X2_train = X2_train
        self.X2_test = X2_test
        self.y_train = y_train
        self.y_test = y_test

    def clean_descriptions(self, descriptions):
        clean_descriptions = descriptions.apply(lambda x: clean_text(x))
        return clean_descriptions

    def train_model(self):
        self.bug_model.construct_model(self.vocab_size, self.embedding_dim, self.max_length, self.dropout)
        self.bug_model.fit_model([self.X1_train, self.X2_train], self.y_train, [self.X1_test, self.X2_test], self.y_test, self.num_epochs)

    def plot_graphs(self):
        self.bug_model.plot_graphs()

    def save_model(self):
        self.bug_model.save_model()

    def load_model(self):
        self.bug_model.load_model()

    def predict(self, descriptions1, descriptions2):
        descriptions1 = np.array(text_to_padded(self.clean_descriptions(descriptions1), self.tokenizer, self.max_length))
        descriptions2 = np.array(text_to_padded(self.clean_descriptions(descriptions2), self.tokenizer, self.max_length))
        return self.bug_model.predict([descriptions1, descriptions2])

        




    