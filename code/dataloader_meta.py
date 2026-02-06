import pandas as pd
import gzip
from modelscope import AutoTokenizer, AutoModel
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
suffix = "Compress this item's feature into one word: [EMB]"
for i, row in df_remap.iterrows():
    sen = row['title'] + ' ' + row['brand'] + ' '
    # cates = eval(row['categories'])
    cates = (row['categories'])
    if isinstance(cates, list):
        for c in cates[0]:
            sen = sen + c + ' '
    sen += row['description']
    sen = sen.replace('\n', ' ')
    sen += ' ' + suffix
    sentences.append(sen)

print(sentences[:10])

# 加载 Llama 模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.add_tokens(['[EMB]'])
model.resize_token_embeddings(len(tokenizer))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print('model load done!')

# 对文本进行编码
sentence_embeddings = []
batch_size = 16
for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i + batch_size]
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, add_special_token=False).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, -1, :].cpu().numpy()
        sentence_embeddings.extend(embeddings)

sentence_embeddings = np.array(sentence_embeddings)
print('text encoded!')

assert sentence_embeddings.shape[0] == df_remap.shape[0]
np.save(os.path.join('D:/pycharm_project/LightGCN-PyTorch-master/data/Beauty/Beauty', 'text_feat_llama.npy'),
        sentence_embeddings)
print('done!')

print(sentence_embeddings[:10])

load_txt_feat = np.load('D:/pycharm_project/LightGCN-PyTorch-master/data/Beauty/Beauty/text_feat_llama.npy',
                        allow_pickle=True)
print(load_txt_feat.shape)
print(load_txt_feat[:10])
