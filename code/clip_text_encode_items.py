import pandas as pd
import gzip
from modelscope import AutoModel, AutoProcessor
import torch
import os
import numpy as np


# 抽取并保存文本模态特征

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


df = getDF('D:/pycharm_project/LightGCN-PyTorch-master/data/Beauty/Beauty/meta_Beauty.json.gz')
df_remap = pd.read_csv('D:/pycharm_project/LightGCN-PyTorch-master/data/Beauty/Beauty/remap_item.csv')

df_remap = df_remap.merge(df, on='asin')

# print(df.head())
print(df_remap.iloc[0])

headers = df_remap.columns.tolist()
print(headers)

print(len(df_remap))

# sentences: title + brand + category + description | All have title + description

title_na_df = df_remap[df_remap['title'].isnull()]
print(title_na_df.shape)

desc_na_df = df_remap[df_remap['description'].isnull()]
print(desc_na_df.shape)

na_df = df_remap[df_remap['description'].isnull() & df_remap['title'].isnull()]
print(na_df.shape)

na3_df = df_remap[df_remap['description'].isnull() & df_remap['title'].isnull() & df_remap['brand'].isnull()]
print(na3_df.shape)

na4_df = df_remap[df_remap['description'].isnull() & df_remap['title'].isnull() & df_remap['brand'].isnull() & df_remap[
    'categories'].isnull()]
print(na4_df.shape)

df_remap['description'] = df_remap['description'].fillna(" ")
df_remap['title'] = df_remap['title'].fillna(" ")
df_remap['brand'] = df_remap['brand'].fillna(" ")
df_remap['categories'] = df_remap['categories'].fillna(" ")

sentences = []
for i, row in df_remap.iterrows():
    sen = row['title'] + ' ' + row['brand'] + ' '
    # cates = eval(row['categories'])
    cates = (row['categories'])
    if isinstance(cates, list):
        for c in cates[0]:
            sen = sen + c + ' '
    sen += row['description']
    sen = sen.replace('\n', ' ')

    sentences.append(sen)

print(sentences[:10])
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "AI-ModelScope/clip-vit-large-patch14"
model = AutoModel.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


print('model load done!')

save_path = os.path.join('D:/pycharm_project/LightGCN-PyTorch-master/data/Beauty/Beauty/', 'text_feat_clip.npy')

for i in range(len(sentences)):
    inputs = processor(text=sentences[i], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(inputs["input_ids"]).cpu().numpy()

    if i == 0:
        np.save(save_path, text_features)
    else:
        existing_features = np.load(save_path)
        new_features = np.vstack((existing_features, text_features))
        np.save(save_path, new_features)
    print(f"Processed {i}")

print('text encoded!')

load_txt_feat = np.load(save_path, allow_pickle=True)
print(load_txt_feat.shape)
print(load_txt_feat[:10])