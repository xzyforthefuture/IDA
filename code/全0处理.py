import numpy as np
def replace_zero_rows(image_feats):
    # 找出全零行的索引
    zero_rows = np.all(image_feats == 0, axis=1)
    # 找出非全零行
    non_zero_rows = image_feats[~zero_rows]
    # 计算非全零行的元素平均值
    if non_zero_rows.size > 0:
        avg_row = np.mean(non_zero_rows, axis=0)
        # 将全零行替换为平均值行
        image_feats[zero_rows] = avg_row
    return image_feats


dataset_name = "toys"
item_embedding_path_image = '../data/' + f'{dataset_name}/' + f"{dataset_name}/" + "item_image_features.npy"
item_embedding_path_image_new = '../data/' + f'{dataset_name}/' + f"{dataset_name}/" + "item_image_features_new.npy"
try:
    image_feats = np.load(item_embedding_path_image, allow_pickle=True)
    # 处理全零行
    image_feats = replace_zero_rows(image_feats)
    # 将处理后的 image_feats 重新存入文件
    np.save(item_embedding_path_image_new, image_feats)
    print("处理完成，结果已保存。")
except FileNotFoundError:
    print(f"未找到文件: {item_embedding_path_image}")
except Exception as e:
    print(f"发生错误: {e}")
