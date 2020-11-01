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

# Loading Tokenizer and Encoding our Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)
# after we load the tokenizer we gonna encode the data!
# we will; make train and validation encoded!
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

# saving encoded data into two dataset (dataset_train, dataset_val(test))!
# then let us see the shape of it both!
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

print(f'Train Dataset: {len(dataset_train)}')
print(f'Test Dataset: {len(dataset_val)}')

# Setting up BERT Pretrained Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# Creating Data Loaders
batch_size = 32

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)

# Setting Up Optimizer and Scheduler
# we gonna use Adam as our optimizer!
optimizer = AdamW(model.parameters(),
                  lr=1e-05,
                  eps=1e-12)

epochs = 5

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


