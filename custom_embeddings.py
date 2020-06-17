import numpy as np
import pandas as pd
import os
import re
import time

from gensim.models import Word2Vec
from tqdm import tqdm

from text_preprocessor import clean_text

dataset_path = 'datasets/training_dataset_embeddings.csv'

custom_glove_path = 'datasets/custom_glove_50d.txt'

tqdm.pandas()

df = pd.read_csv(dataset_path)

df['clean_description'] = df['description'].apply(lambda x: clean_text(x))

descriptions = df['clean_description']

train_descriptions = list(descriptions.progress_apply(str.split).values)

start_time = time.time()

model = Word2Vec(sentences=train_descriptions,
                 sg=1,
                 size=50,
                 workers=4)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')

print(len(model.wv.vocab.keys()))

print(model.wv.vector_size)

model.wv.save_word2vec_format(custom_glove_path)


