import world
import utils
from world import cprint
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
def count_item_interaction(file_path, target_item_id):
    interaction_count = 0
    with open(file_path, "r") as file:
        for line in file:
            items = line.strip().split()
            for item in items[1:]:
                if item == str(int(target_item_id)):
                    interaction_count += 1
    return interaction_count
Recmodel = register.MODELS[world.model_name](world.config, dataset)
if torch.cuda.device_count() > 1:
    Recmodel = torch.nn.DataParallel(Recmodel, [0, 1])
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel.module, world.config)
crossentropyloss = utils.CrossEntropyLoss(Recmodel, world.config)
kldivergence = utils.KLDivergenceLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best = 0
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            recall20, user_embedding, item_embedding, rating_list = Procedure.Test(dataset, Recmodel.module, epoch, w, world.config['multicore'])
            if recall20 > best:
                best = recall20
                user_embedding_np = user_embedding.detach().cpu().numpy()
                item_embedding_np = item_embedding.detach().cpu().numpy()
                # np.save(
                #     'D:/pycharm_project/LightGCN-PyTorch-master/data/' + world.dataset + '/' + world.dataset + '/' + 'My_' + world.dataset + '_user_embeddings.npy',
                #     user_embedding_np)
                # np.save(
                #     'D:/pycharm_project/LightGCN-PyTorch-master/data/' + world.dataset + '/' + world.dataset + '/' + 'My_' + world.dataset + '_user_embeddings.npy',
                #     item_embedding_np)
                #
                # with open(
                #         'D:/pycharm_project/LightGCN-PyTorch-master/data/' + world.dataset + '/' + world.dataset + '/' + world.dataset + '_' + 'LAIC' + '.txt',
                #         mode='w') as f:
                #     for i in rating_list:
                #         for j in i.numpy():
                #             for k in j:
                #                 f.write(str(k) + ', ')
        output_information = Procedure.BPR_train_original(dataset, Recmodel.module, bpr, epoch, neg_k=Neg_k,w=w)
        end = time.time()
        epoch_time = end - start
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        if (epoch + 1) % 10 == 0:
            print(f"Total time for last 10 epochs: {epoch_time * 10:.2f} seconds")


finally:
    if world.tensorboard:
        w.close()

# α1=0.2， τ1=0.15