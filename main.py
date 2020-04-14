from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from text_preprocessor import clean_text, text_to_padded
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

oov_token = '<OOV>'
vocab_size = 10000
embedding_dim = 16
training_portion = 0.8
max_length = 800

data_full = pd.read_csv('datasets/training_dataset.csv', sep=',')

# smaller dataset for testing
data_full = data_full[:100000]

data_full['cleaned_description'] = data_full['description'].apply(lambda x: clean_text(x))

training_size = int(len(data_full.index) * training_portion)

training_descriptions = data_full['cleaned_description'][0:training_size]
training_labels = data_full['is_bug'][0:training_size]
testing_descriptions = data_full['cleaned_description'][training_size:]
testing_labels = data_full['is_bug'][training_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_descriptions)
word_index = tokenizer.word_index

training_padded = text_to_padded(training_descriptions, tokenizer, max_length)
testing_padded = text_to_padded(testing_descriptions, tokenizer, max_length)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()