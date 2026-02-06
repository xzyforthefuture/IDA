import pandas as pd
import gzip
import collections
from tqdm import tqdm

# 数据集读取，用户、物品ID重映射，数据集划分

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#############   Reading dataset  #############
df = getDF('reviews_Baby_5.json.gz')

print(df.head())

headers = df.columns.tolist()
print(headers)

lenth = len(df)
print(len(df))

print(df['reviewerID'].value_counts())
print(df['asin'].value_counts())

############     Remapping users & items  ############
user_list = []
item_list = []
for i in tqdm(range(lenth)):
  inter = df.iloc[i]
  user = inter[0]
  item = inter[1]
  user_list.append(user)
  item_list.append(item)

users = list(set(user_list))
items = list(set(item_list))
users.sort(key=user_list.index)
items.sort(key=item_list.index)

user_id = []
item_id = []
for i in range(0, len(users), 1):
  user_id.append(int(i))
for i in range(0, len(items), 1):
  item_id.append(int(i))

print(len(user_id))
print(len(item_id))

df_remap_user = pd.DataFrame()
df_remap_user['reviewerID'] = users
df_remap_user['user_id'] = user_id

df_remap_item = pd.DataFrame()
df_remap_item['asin'] = items
df_remap_item['item_id'] = item_id

print(df_remap_user.head())
print(df_remap_item.head())
df_remap_item.to_csv("remap_item.csv", index=False)

df = df.merge(df_remap_user, on='reviewerID')
df = df.merge(df_remap_item, on='asin')

print(df.head())

df = df.drop(columns=["reviewerName", "helpful", "reviewText", "overall", "summary", "unixReviewTime", "reviewTime"])

print(df.head())
df.to_csv("interaction.csv", index=False)

####################  interaction ####################
inter_dict = collections.defaultdict(list)
for i in tqdm(range(lenth)):
  inter = df.iloc[i]
  user  = inter[2]
  item  = inter[3]
  inter_dict[user].append(item)

print(len(inter_dict))
###################   train and test split ################
import numpy as np

train_dict = collections.defaultdict(list)
test_dict = collections.defaultdict(list)

for i in tqdm(range(len(df_remap_user))):
  items = inter_dict[i]
  lenth = len(items)
  test_idx = np.random.choice(list(range(lenth)), size=int(lenth * 0.2), replace=False)
  test_list = []
  for idx in test_idx:
    item = items[idx]
    test_list.append(item)
  test_dict[i] = test_list
  train_dict[i] = list(set(items) - set(test_list))

# 保存字典
import pickle

with open("train.pickle", "wb") as f:
    pickle.dump(train_dict, f)

with open("test.pickle", "wb") as f:
    pickle.dump(test_dict, f)





