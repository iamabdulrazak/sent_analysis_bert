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
# creating a new label column
possible_labels = df.category.unique()
label_dict = {}

for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

df['label'] = df.category.replace(label_dict)
df.head()

# Training/Validation Split
# spliting data into train and validation!
X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values,
				 		  test_size=0.15, random_state=17,
						  stratify=df.label.values)
# creating a data_type comlumn!
# to see how much of the dataset goes to the train and test(validation)!
df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

df.groupby(['category', 'label', 'data_type']).count()

