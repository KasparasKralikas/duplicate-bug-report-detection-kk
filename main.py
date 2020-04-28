from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from text_preprocessor import clean_text, text_to_padded
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from bug_model import BugModel

from bug_model_client import BugModelClient

bug_model_client = BugModelClient()
bug_model_client.init_data(14000)
#bug_model_client.train_model()
bug_model_client.load_model()
print(bug_model_client.data['description'][:10])
print(bug_model_client.predict(bug_model_client.data['description'][:10], 10))