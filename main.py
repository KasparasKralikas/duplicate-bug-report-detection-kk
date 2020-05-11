from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from text_preprocessor import clean_text, text_to_padded
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import io
from bug_model import BugModel
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split

from bug_model_client import BugModelClient

import time


bug_model_client = BugModelClient()
bug_model_client.init_data(19000)
bug_model_client.prepare_embedding()
bug_model_client.train_model()
bug_model_client.plot_graphs()
bug_model_client.save_model()
bug_model_client.load_model()

data = pd.read_csv('datasets/bugs_dataset.csv', sep=',')

all_bugs = data[:len(data)-3]
new_bugs = data[-3:]
print(new_bugs)


start = time.time()
print(bug_model_client.predict_top_k(new_bugs['description'], all_bugs['description'], all_bugs['case_id'], all_bugs['master_id_label'], 20))
end = time.time()
print(end - start)

'''

# enabling the pretrained model for trainig our custom model using tensorflow hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.KerasLayer(module_url)
print(embed)

# creating a method for embedding and will using method for every input layer 
def UniversalEmbedding(x):
    print(tf.squeeze(tf.cast(x, tf.string)))
    return embed(tf.squeeze(tf.cast(x, tf.string)))

DROPOUT = 0.1

df = pd.read_csv('datasets/training_dataset_pairs.csv', sep=',')[:2000]

# Taking the question1 as input and ceating a embedding for each question before feed it to neural network
q1 = layers.Input(shape=(1,), dtype=tf.string)
embedding_q1 = layers.Lambda(UniversalEmbedding, output_shape=(512,))(q1)
# Taking the question2 and doing the same thing mentioned above, using the lambda function
q2 = layers.Input(shape=(1,), dtype=tf.string)
embedding_q2 = layers.Lambda(UniversalEmbedding, output_shape=(512,))(q2)

# Concatenating the both input layer
merged = layers.concatenate([embedding_q1, embedding_q2])
merged = layers.Dense(200, activation='relu')(merged)
merged = layers.Dropout(DROPOUT)(merged)

# Normalizing the input layer,applying dense and dropout  layer for fully connected model and to avoid overfitting 
merged = layers.BatchNormalization()(merged)
merged = layers.Dense(200, activation='relu')(merged)
merged = layers.Dropout(DROPOUT)(merged)

merged = layers.BatchNormalization()(merged)
merged = layers.Dense(200, activation='relu')(merged)
merged = layers.Dropout(DROPOUT)(merged)

merged = layers.BatchNormalization()(merged)
merged = layers.Dense(200, activation='relu')(merged)
merged = layers.Dropout(DROPOUT)(merged)

# Using the Sigmoid as the activation function and binary crossentropy for binary classifcation as 0 or 1
merged = layers.BatchNormalization()(merged)
pred = layers.Dense(2, activation='sigmoid')(merged)
model = Model(inputs=[q1,q2], outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

from sklearn.model_selection import train_test_split

X1 = df['description_1']
X2 = df['description_2']
y = df['duplicates']

X1_train, X1_test,X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

train_q1 = X1_train.tolist()
train_q1 = np.array(train_q1, dtype=object)[:, np.newaxis]
train_q2 = X2_train.tolist()
train_q2 = np.array(train_q2, dtype=object)[:, np.newaxis]

train_labels = np.asarray(pd.get_dummies(y_train), dtype = np.int8)

test_q1 = X1_test.tolist()
test_q1 = np.array(test_q1, dtype=object)[:, np.newaxis]
test_q2 = X2_test.tolist()
test_q2 = np.array(test_q2, dtype=object)[:, np.newaxis]

test_labels = np.asarray(pd.get_dummies(y_test), dtype = np.int8)

history = model.fit([train_q1, train_q2], 
            train_labels,
            validation_data=([test_q1, test_q2], test_labels),
            epochs=10,
            batch_size=512, verbose=2)

'''