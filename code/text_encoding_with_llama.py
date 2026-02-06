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


df = getDF('D:/pycharm_project/LAIC-main/data/Beauty/Beauty/meta_Beauty.json.gz')
df_remap = pd.read_csv('D:/pycharm_project/LAIC-main/data/Beauty/Beauty/remap_item.csv')

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

device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载 Llama 模型和分词器
model = AutoModel.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map=device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", padding_side='left')

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))

print('model load done!')

# 可以根据实际情况进一步减小批量大小
batch_size = 1
save_path = os.path.join('D:/pycharm_project/LightGCN-PyTorch-master/data/cellphones/cellphones/', 'text_feat_llama.npy')
first_batch = True

for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i + batch_size]
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, -1, :].cpu().numpy()

    # 释放GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if first_batch:
        np.save(save_path, embeddings)
        first_batch = False
    else:
        existing_embeddings = np.load(save_path, allow_pickle=True)
        combined_embeddings = np.vstack((existing_embeddings, embeddings))
        # np.save(save_path, combined_embeddings)

print('text encoded!')

load_txt_feat = np.load(save_path, allow_pickle=True)
print(load_txt_feat.shape)
print(load_txt_feat[:10])