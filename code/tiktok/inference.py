import os
import argparse
import time
import copy
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.set_num_threads(2)

seed = 2020
torch.manual_seed(seed) # cpu
torch.cuda.manual_seed(seed) #gpu
np.random.seed(seed) #numpy
random.seed(seed) #random and transforms
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MMGCN', help='Model name.')
    parser.add_argument('--data_path', default='../../../data/tiktok/', help='Dataset path')
    parser.add_argument('--save_path', default='./model_1/', help='saved model path')
    parser.add_argument('--log_name', default='adressa', help='training log name')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--early_stop', type=int, default=10, help='early_stop Epoch number.')
    parser.add_argument('--num_workers', type=int, default=2, help='Workers number.')
    parser.add_argument('--num_user', type=int, default=31123, help='User number.')
    parser.add_argument('--num_item', type=int, default=4895, help='Item number.')
    parser.add_argument('--aggr_mode', default='mean', help='Aggregation mode.')
    parser.add_argument('--num_layer', type=int, default=2, help='Layer number.')
    parser.add_argument('--has_id', type=bool, default=True, help='Has id_embedding')
    parser.add_argument('--concat', type=bool, default=False, help='Concatenation')
    args = parser.parse_args()
    print("arguments: %s " %(args))
   ###############################################################################################################################
    print('Data loading ...')
    test_dataset = np.load(args.data_path+'test_full.npy', allow_pickle=True)
    print('Data has been loaded.')
###############################################################################################################################
    model = torch.load('{}{}_{}.pth'.format(args.save_path, args.model_name, args.log_name))
    model.eval()
    with torch.no_grad():
        test_precision, test_recall, test_ndcg_score = model.full_ranking(test_dataset, topk=[10, 20, 50, 100])
        print('Precision: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_precision[0], test_precision[1], test_precision[2], test_precision[3]))
        print('Recall: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_recall[0], test_recall[1], test_recall[2], test_recall[3]))
        print('NDCG: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_ndcg_score[0], test_ndcg_score[1], test_ndcg_score[2], test_ndcg_score[3]))
