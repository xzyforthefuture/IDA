import pickle
import numpy as np


class SomeClass:
    def __init__(self):
        self.m_item = 0
        self.n_user = 0
        self.traindataSize = 0
        trainUniqueUsers = []
        trainUser = []
        trainItem = []

        train_file = 'D:/pycharm_project/LightGCN-PyTorch-master/data/toys/toys/train.pickle'
        item_embedding_file = 'D:/pycharm_project/LightGCN-PyTorch-master/data/toys/toys/item_image_features_new.npy'
        user_embedding_file = 'D:/pycharm_project/LightGCN-PyTorch-master/data/toys/toys/user_image_features_new.npy'

        # 读取物品的 embedding
        item_embeddings = np.load(item_embedding_file, allow_pickle=True)

        # 读取 pickle 文件
        with open(train_file, 'rb') as f:
            data_dict = pickle.load(f)

        user_embeddings = {}
        for uid, items in data_dict.items():
            uid = int(uid)
            items = [int(i) for i in items]
            trainUniqueUsers.append(uid)
            trainUser.extend([uid] * len(items))
            trainItem.extend(items)
            self.m_item = max(self.m_item, max(items))
            self.n_user = max(self.n_user, uid)
            self.traindataSize += len(items)

            # 计算用户的 embedding
            user_item_embeddings = item_embeddings[items]
            user_embedding = np.mean(user_item_embeddings, axis=0)
            user_embeddings[uid] = user_embedding

        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # 将用户的 embedding 存储为 numpy 数组
        all_user_embeddings = np.array([user_embeddings[uid] for uid in sorted(user_embeddings.keys())])

        # 保存用户的 embedding 到文件
        np.save(user_embedding_file, all_user_embeddings)


# 示例使用
if __name__ == "__main__":
    obj = SomeClass()
    print(obj.trainUniqueUsers)
    print(obj.trainUser)
    print(obj.trainItem)
    print(obj.m_item)
    print(obj.n_user)
    print(obj.traindataSize)
