import world
# import dataloader
import dataloader_new as dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'citeulikea-1m', 'bookcrossing', 'movielens', 'ml1m', "last-fm", "Baby", "Beauty", "toys", "cellphones"]:
    dataset = dataloader.Loader(path="../data/"+ world.dataset + "/" + world.dataset)
    # dataset = dataloader.Loader(path="../data/"+ world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'sgl': model.SGL,
    'ma': model.Modal_Alignment
}