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
    vocab_size = None
    embedding_dim = 50
    training_portion = 0.8
    max_length = 800
    num_epochs = 3
    dropout = 0.1

    data_path = 'datasets/training_dataset_pairs.csv'

    custom_glove_path = 'datasets/custom_glove_50d.txt'

    data = None

    training_size = None

    word_index = None

    tokenizer = None

    embedding_matrix = None

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
        self.vocab_size = len(self.word_index) + 1

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

    def prepare_embedding(self):
        embeddings_index = dict()
        f = open(self.custom_glove_path, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        embeddings_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector
        self.embedding_matrix = embeddings_matrix

    def clean_descriptions(self, descriptions):
        clean_descriptions = descriptions.apply(lambda x: clean_text(x))
        return clean_descriptions

    def train_model(self):
        self.bug_model.construct_model(self.vocab_size, self.embedding_dim, self.max_length, self.dropout, self.embedding_matrix)
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

    def predict_top_k(self, descriptions, all_descriptions, labels, k):
        descriptions = np.array(text_to_padded(self.clean_descriptions(descriptions), self.tokenizer, self.max_length))
        all_descriptions = np.array(text_to_padded(self.clean_descriptions(all_descriptions), self.tokenizer, self.max_length))
        all_predictions = []
        for description in descriptions:
            description_repeated = np.full((len(all_descriptions), self.max_length), description)
            predictions = self.bug_model.predict([description_repeated, all_descriptions])
            predictions = np.array([prediction[0] for prediction in predictions])
            predictions_top_k_indices = (-predictions).argsort()[:k]
            prediction_summary = []
            for index in predictions_top_k_indices:
                prediction_summary.append({'case_id': labels[index], 'probability': predictions[index]})
            all_predictions.append(prediction_summary)
        return all_predictions



        




    