import requests
import pandas as pd
import gzip
from tqdm import tqdm

def download_images(url, idx):
    # 发送GET请求获取图片内容
    response = requests.get(url)
    path = f'D:/pycharm_project/LightGCN-PyTorch-master/data/Beauty/Beauty/content/{idx}' + '.jpg'''
    # 检查请求是否成功
    if response.status_code == 200:
        # 将图片内容写入文件
        with open(path, 'wb') as file:
            file.write(response.content)
        # print('图片已保存')
    else:
        print('请求图片失败')
        print(idx)

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

headers = df_remap.columns.tolist()
print(headers)

lenth = len(df_remap)
print(lenth)

for i in tqdm(range(lenth)):
    url = df_remap['imUrl'][i]
    try:
        download_images(url, i)
    except Exception:
        continue

