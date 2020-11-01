# we start implementing libraries
# and check if there is any issues wih it!!
# used try/except to handle that!
try:
  import pandas as pd
  import numpy as np
  import random
  import torch

  from tqdm.notebook import tqdm
  from sklearn.model_selection import train_test_split
  from transformers import BertTokenizer
  from torch.utils.data import TensorDataset
  from transformers import BertForSequenceClassification
  from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
  from transformers import AdamW, get_linear_schedule_with_warmup
  from sklearn.metrics import f1_score

except Exception as e:
  print('Packages are Missing! \n {}'.format(e))


# Exploratory Data Analysis and Preprocessing
# Small Twitter Dataset!!
df = pd.read_csv('data/smile.csv', names=['id', 'text', 'category'])
df.set_index('id', inplace=True)
df.head()
# seeing the total values in each label!
df.category.value_counts()
# cleaning the un-wanted labels!
# we delete nocode because we saw it as a wasted data lable!
# but, for the multi-lables we delete it because it will make the training more complex!
# because the dataset is small!!
df = df[~df.category.str.contains('\|')]
df = df[df.category != 'nocode']
df.category.value_counts()
#
