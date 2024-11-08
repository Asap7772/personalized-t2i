import pandas as pd
import datasets
import os
os.environ['HF_TOKEN'] = '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'

print('Loading dataset...') 
df = pd.read_json('/home/anikait.singh/personalized-t2i/data/train.json')
print('Dataset loaded!')
df = df.groupby('user_id').filter(lambda x: len(x) >= 10)
print('Number of unique users:', len(df['user_id'].unique()))

# create classification dataset emb -> user_id
unique_users = df['user_id'].unique()
user2emb = {user: i for i, user in enumerate(unique_users)}
df['class'] = df['user_id'].map(user2emb)

# split dataset into train and test
train_df = df.sample(frac=0.9, random_state=0)
test_df = df.drop(train_df.index)

train_ds = datasets.Dataset.from_pandas(train_df)
test_ds = datasets.Dataset.from_pandas(test_df)
ds_dict = {'train': train_ds, 'test': test_ds}
ds_dict = datasets.DatasetDict(ds_dict)
ds_dict.push_to_hub('Asap7772/emb_classify')