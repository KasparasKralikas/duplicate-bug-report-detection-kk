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
bug_model_client.init_data(30000)
bug_model_client.prepare_embedding()
bug_model_client.train_model()
bug_model_client.save_model()
bug_model_client.plot_graphs()
bug_model_client.load_model()

all_bugs = pd.read_csv('datasets/bugs_dataset.csv', sep=',')

new_bugs = pd.read_csv('datasets/bugs_dataset_testing.csv', sep=',')[:500]
new_bugs.reset_index(inplace=True)

start = time.time()
bug_model_client.validate_predict_top_k(new_bugs['description'], new_bugs['case_id'], new_bugs['master_id_label'], all_bugs['description'], all_bugs['case_id'], all_bugs['master_id_label'], 20)
end = time.time()
print(end - start)
